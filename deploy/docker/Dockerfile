# ─────────────────────────────────────────────────────────────
# ⚙️ Builder Stage
# ─────────────────────────────────────────────────────────────
FROM golang:1.23 AS builder

WORKDIR /app

# Copy go module files and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source
COPY . .

# Build the Go binary (assuming main.go is at cmd/server/main.go)
RUN go build -o sematic-cache ./cmd/server/main.go

# ─────────────────────────────────────────────────────────────
# 🧼 Final Runtime Image
# ─────────────────────────────────────────────────────────────
FROM alpine:latest

# Set working directory for runtime
WORKDIR /app

# Install any needed OS packages (optional, minimal base)
# RUN apk add --no-cache ca-certificates

# Copy binary from the builder
COPY --from=builder /app/sematic-cache .

# Make sure it’s executable
RUN chmod +x /app/sematic-cache

# Set default entrypoint
ENTRYPOINT ["/app/sematic-cache"]