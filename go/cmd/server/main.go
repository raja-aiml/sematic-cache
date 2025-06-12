// Binary server runs the cache server.
package main

import (
	"flag"
	"github.com/raja-aiml/sematic-cache/go/core"
	"github.com/raja-aiml/sematic-cache/go/server"
)

func main() {
	addr := flag.String("address", ":8080", "server address")
	flag.Parse()

	cache := core.NewCache(100)
	srv := server.New(cache)
	srv.Run(*addr)
}
