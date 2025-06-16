# ─────────────────────────────────────────────────────────────
# ⚙️ Builder Stage
# ─────────────────────────────────────────────────────────────
FROM golang:1.23 AS builder

WORKDIR /app

# Copy module files and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application code
COPY . .

# Build the binary from the root (expects main.go in /app)
RUN go build -o sematic-cache cmd/server/main.go 

# ─────────────────────────────────────────────────────────────
# 🧼 Final Runtime Image
# ─────────────────────────────────────────────────────────────
FROM alpine:latest

WORKDIR /bin

# Copy binary from builder
COPY --from=builder /app/sematic-cache .

# Run binary
ENTRYPOINT ["/bin/sematic-cache"]