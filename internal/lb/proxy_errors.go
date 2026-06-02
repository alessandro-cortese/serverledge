package lb

import (
	"fmt"
	"net/http"

	"github.com/labstack/echo/v4"
)

const proxyErrorStatusKey = "serverledge.proxy_error_status"
const proxyErrorMessageKey = "serverledge.proxy_error_message"

func setProxyError(c echo.Context, status int, message string) {
	c.Set(proxyErrorStatusKey, status)
	c.Set(proxyErrorMessageKey, message)
}

func setHardwareNotSupported(c echo.Context, functionName string) {
	message := "Hardware not supported"
	if functionName != "" {
		message = fmt.Sprintf("Hardware not supported for function '%s'", functionName)
	}

	setProxyError(c, http.StatusTooManyRequests, message)
}

// proxyErrorMiddleware is an Echo middleware that intercepts proxy errors set by Next()
// and converts them into proper HTTP responses before they reach the client.
//
// Echo's proxy middleware calls Next() to select a target node. When Next()
// returns nil (no compatible node available), Echo responds with 502 Bad Gateway by default.
// To return a more meaningful status code (429 for hardware not supported), Next()
// stores the desired status and message in the Echo context via setProxyError().
// This middleware then reads those values and overwrites the response accordingly.
//
// The middleware also handles panics that may occur inside the proxy (refused
// connections): if a proxy error was previously set in the context, the panic is
// caught and converted into the appropriate HTTP response instead of propagating.
func proxyErrorMiddleware(next echo.HandlerFunc) echo.HandlerFunc {
	return func(c echo.Context) (err error) {

		// Deferred panic recovery: catches any panic that occurs during proxy execution.
		// If a proxy error was previously set in the context by Next() (via setProxyError),
		// it is converted into a structured HTTP response.
		// If no proxy error was set, the panic is re-propagated so it is not silently swallowed.
		defer func() {
			if recovered := recover(); recovered != nil {
				status, ok := c.Get(proxyErrorStatusKey).(int)
				if !ok {
					// No proxy error was set: this is an unexpected panic, re-propagate it.
					panic(recovered)
				}

				// The response may have already been partially sent to the client
				// (headers already flushed). In that case we cannot overwrite it,
				// so we silently discard the error to avoid a double-write panic.
				if c.Response().Committed {
					err = nil
					return
				}

				// Write the structured error response using the status stored in the context.
				err = writeProxyError(c, status)
			}
		}()

		// Execute the proxy middleware chain, which internally calls Next() to select
		// a target node and forwards the request to it.
		err = next(c)

		// After the proxy has executed, check whether Next() stored a proxy error
		// in the context. This happens when no compatible node was found.
		status, ok := c.Get(proxyErrorStatusKey).(int)
		if !ok {
			// No proxy error was set: the request was handled normally, return as-is.
			return err
		}

		// Same committed check as in the deferred recovery: if Echo has already sent
		// the response headers to the client, we cannot overwrite the status code.
		if c.Response().Committed {
			return err
		}

		// Overwrite the default proxy response (502) with the structured error response
		// containing the status code and message set by Next().
		return writeProxyError(c, status)
	}
}

func writeProxyError(c echo.Context, status int) error {
	message, _ := c.Get(proxyErrorMessageKey).(string)
	if message == "" {
		message = http.StatusText(status)
	}

	return c.JSON(status, map[string]interface{}{
		"Success": false,
		"Error":   message,
	})
}
