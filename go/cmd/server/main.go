// Binary server runs the cache server.
package main

import (
	"context"
	"flag"
	"log"
	"os"

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
	srv.Run(*addr)
}
