package observability

import (
	"github.com/raja-aiml/sematic-cache/internal/logger"
)

// SetupLogging configures structured logging with zap
func SetupLogging() {
	logger.Initialize()
}
