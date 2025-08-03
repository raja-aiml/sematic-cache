# ─────────────────────────────────────────────────────────────
# ⚙️ Builder Stage
# ─────────────────────────────────────────────────────────────
FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git file

WORKDIR /app

# Copy go module files first (for better caching)
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Verify source files exist
RUN ls -la main.go cmd/

# Build with verbose output and static linking
RUN CGO_ENABLED=0 GOOS=linux go build \
    -a -installsuffix cgo \
    -ldflags '-extldflags "-static"' \
    -o sematic-cache \
    .

# Verify the binary was created
RUN ls -la sematic-cache && echo "Binary created successfully"

# ─────────────────────────────────────────────────────────────
# 🧼 Final Runtime Image  
# ─────────────────────────────────────────────────────────────
FROM alpine:latest

# Install ca-certificates for HTTPS calls
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/sematic-cache .

# Verify the binary exists and is executable
RUN ls -la /app/sematic-cache && \
    chmod +x /app/sematic-cache && \
    echo "Binary verification complete"

# Expose the port
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["/app/sematic-cache"]