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

func proxyErrorMiddleware(next echo.HandlerFunc) echo.HandlerFunc {
	return func(c echo.Context) (err error) {
		defer func() {
			if recovered := recover(); recovered != nil {
				status, ok := c.Get(proxyErrorStatusKey).(int)
				if !ok {
					panic(recovered)
				}

				if c.Response().Committed {
					err = nil
					return
				}

				err = writeProxyError(c, status)
			}
		}()

		err = next(c)

		status, ok := c.Get(proxyErrorStatusKey).(int)
		if !ok {
			return err
		}

		if c.Response().Committed {
			return err
		}

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
