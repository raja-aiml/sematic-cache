FROM golang:1.23 AS builder
WORKDIR /app
COPY .. .
RUN go build -o server ./cmd/server

FROM debian:stable-slim
WORKDIR /app
COPY --from=builder /app/server /app/server
EXPOSE 8080
CMD ["/app/server"]
