// Binary server runs the cache server.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/raja-aiml/sematic-cache/go/core"
	"github.com/raja-aiml/sematic-cache/go/observability"
	"github.com/raja-aiml/sematic-cache/go/server"
)

func main() {
	addr := flag.String("address", ":8080", "server address")
	flag.Parse()

	shutdown, err := observability.Init(context.Background(), "cache-server", os.Getenv("JAEGER_ENDPOINT"))
	if err != nil {
		log.Fatalf("otel init: %v", err)
	}
	defer shutdown(context.Background())

	cache := core.NewCache(100)
	srv := server.New(cache)
	httpSrv := &http.Server{
		Addr:         *addr,
		Handler:      srv,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in background
	go func() {
		log.Printf("server listening on %s", *addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}
}
