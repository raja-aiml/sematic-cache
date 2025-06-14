package server

import (
    "net/http"
    "strings"

    "github.com/gin-gonic/gin"
)

// AdminAuth returns a Gin middleware that enforces a bearer token or X-Admin-Token header.
// If token is empty, the middleware allows all requests.
func AdminAuth(token string) gin.HandlerFunc {
    return func(c *gin.Context) {
        if token == "" {
            c.Next()
            return
        }
        // Check Authorization: Bearer <token>
        authHeader := c.GetHeader("Authorization")
        if strings.HasPrefix(authHeader, "Bearer ") {
            if authHeader[len("Bearer "):] == token {
                c.Next()
                return
            }
        }
        // Check X-Admin-Token header
        if c.GetHeader("X-Admin-Token") == token {
            c.Next()
            return
        }
        c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
        c.Abort()
    }
}