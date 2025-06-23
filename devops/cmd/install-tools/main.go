package main

import (
	"context"
	"flag"
	"os"

	"github.com/raja-aiml/sematic-cache/devops/internal/logger"
	"github.com/raja-aiml/sematic-cache/devops/internal/tools"
)

func main() {
	var skipConfirm bool
	flag.BoolVar(&skipConfirm, "skip-confirmation", false, "Skip installation confirmation")
	flag.Parse()

	log := logger.New()
	installer := tools.NewInstaller(skipConfirm)

	ctx := context.Background()
	if err := installer.InstallAll(ctx); err != nil {
		log.Error("Installation failed: %v", err)
		os.Exit(1)
	}
}
