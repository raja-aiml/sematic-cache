# Build stage
FROM golang:1.21.6-alpine3.20 AS builder

# Install dependencies
RUN apk update && apk upgrade && apk add --no-cache git make

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=$(git describe --tags --always) \
    -X main.buildTime=$(date -u '+%Y-%m-%d_%H:%M:%S') \
    -X main.gitCommit=$(git rev-parse HEAD)" \
    -o bin/iaac .

# Runtime stage
FROM alpine:3.20

# Install runtime dependencies
RUN apk update && apk upgrade && apk add --no-cache \
    ca-certificates \
    curl \
    bash \
    kubectl \
    docker-cli

# Install k3d
RUN curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Create non-root user
RUN addgroup -g 1000 deploy && \
    adduser -D -u 1000 -G deploy deploy

# Copy binary from builder
COPY --from=builder /app/bin/iaac /usr/local/bin/

# Set permissions
RUN chmod +x /usr/local/bin/iaac

# Switch to non-root user
USER deploy

# Set entrypoint
ENTRYPOINT ["iaac"]