package reduction

import (
	"fmt"

	"github.com/raja-aiml/sematic-cache/internal/logger"
)

// ReducerType represents the type of dimension reduction algorithm
type ReducerType string

const (
	// PCAReducerType represents Principal Component Analysis
	PCAReducerType ReducerType = "pca"
	// PCAGonumReducerType represents Gonum-optimized PCA
	PCAGonumReducerType ReducerType = "pca_gonum"
	// IncrementalPCAReducerType represents Incremental PCA
	IncrementalPCAReducerType ReducerType = "incremental_pca"
)

// ReducerFactory creates dimension reduction algorithms
type ReducerFactory struct{}

// NewReducerFactory creates a new reducer factory
func NewReducerFactory() *ReducerFactory {
	return &ReducerFactory{}
}

// CreateReducer creates a reducer based on the specified type and configuration
func (f *ReducerFactory) CreateReducer(reducerType ReducerType, config ReducerConfig) (Reducer, error) {
	logger.Debug("Creating reducer", logger.Fields{
		"type":       string(reducerType),
		"output_dim": config.OutputDimensions,
		"variance":   config.VarianceRetained,
	})

	switch reducerType {
	case PCAReducerType:
		// Convert ReducerConfig to Config
		pcaConfig := &Config{
			TargetDim:         config.OutputDimensions,
			VarianceThreshold: config.VarianceRetained,
		}
		logger.Info("Created PCA reducer", logger.Fields{
			"target_dim": config.OutputDimensions,
		})
		return NewPCAReducer(pcaConfig), nil
	case PCAGonumReducerType:
		// Convert ReducerConfig to Config
		pcaConfig := &Config{
			TargetDim:         config.OutputDimensions,
			VarianceThreshold: config.VarianceRetained,
		}
		logger.Info("Created PCA Gonum reducer", logger.Fields{
			"target_dim": config.OutputDimensions,
		})
		return NewPCAGonumReducer(pcaConfig), nil
	case IncrementalPCAReducerType:
		// Convert ReducerConfig to Config for IncrementalPCA
		pcaConfig := &Config{
			TargetDim:         config.OutputDimensions,
			VarianceThreshold: config.VarianceRetained,
		}
		logger.Info("Created Incremental PCA reducer", logger.Fields{
			"target_dim": config.OutputDimensions,
			"batch_size": 100,
		})
		return NewIncrementalPCAReducer(pcaConfig, 100), nil
	default:
		err := fmt.Errorf("unknown reducer type: %s", reducerType)
		logger.Error("Failed to create reducer", logger.Fields{
			"type":  string(reducerType),
			"error": err.Error(),
		})
		return nil, err
	}
}

// GetAvailableReducers returns a list of available reducer types
func (f *ReducerFactory) GetAvailableReducers() []ReducerType {
	return []ReducerType{
		PCAReducerType,
		PCAGonumReducerType,
		IncrementalPCAReducerType,
	}
}

// DimensionReducerConfig extends ReducerConfig with factory configuration
type DimensionReducerConfig struct {
	ReducerConfig
	// Type specifies which reducer algorithm to use
	Type ReducerType
	// EnableOptimization uses optimized implementations when available
	EnableOptimization bool
}

// NewDimensionReducerWithFactory creates a dimension reducer using the factory pattern
func NewDimensionReducerWithFactory(config DimensionReducerConfig) (*DimensionReducer, error) {
	factory := NewReducerFactory()

	// Create the appropriate reducer based on configuration
	var reducer Reducer
	var err error

	if config.EnableOptimization && config.Type == PCAReducerType {
		// Use optimized Gonum implementation for PCA if optimization is enabled
		reducer, err = factory.CreateReducer(PCAGonumReducerType, config.ReducerConfig)
	} else {
		reducer, err = factory.CreateReducer(config.Type, config.ReducerConfig)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create reducer: %w", err)
	}

	return &DimensionReducer{
		config:      config.ReducerConfig,
		reducer:     reducer,
		modelLocker: NewModelLocker(),
		metrics:     &QualityMetrics{},
	}, nil
}
