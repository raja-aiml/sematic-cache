FROM golang:1.23 AS builder
WORKDIR /app
COPY . .
# Build the application into a distinct binary name to avoid clashing with the 'server' source directory
RUN mkdir -p bin && \
    go build -o bin/server ./cmd/server

FROM debian:stable-slim
WORKDIR /app
# Copy only the compiled binary from the builder stage (renamed above) into /app/server
COPY --from=builder /app/bin/server /app/server
EXPOSE 8080
CMD ["/app/server"]
