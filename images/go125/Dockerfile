FROM alpine:latest

RUN apk add --no-cache sysbench openssl
COPY images/go125/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

CMD ["/entrypoint.sh"]
