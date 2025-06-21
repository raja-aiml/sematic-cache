# Blueprint Configuration Integration for Config Package

## Overview
The config package has been updated to integrate with the blueprint configuration system, providing dynamic path resolution and scenario management while maintaining backward compatibility.

## Key Changes

### 1. New Fields in AppConfig
- `BlueprintConfig`: Path to a specific blueprint configuration file (optional)
- `PreferredScenario`: User's preferred deployment scenario (optional)

### 2. Blueprint Integration Methods
The Config struct now provides methods to work with the blueprint system:

- `GetBlueprintManager()`: Returns a blueprint manager instance
- `GetScenarioPath(scenario)`: Gets the path to a scenario
- `GetModulePath(module)`: Gets the path to a module
- `GetOverlayPath(overlay)`: Gets the path to an overlay
- `ValidateScenario(scenario)`: Validates if a scenario exists
- `GetScenarioNamespaces(scenario)`: Gets namespaces for a scenario
- `ListAvailableScenarios()`: Lists all available scenarios

### 3. Environment Variables
New environment variables are supported:
- `IAAC_BLUEPRINT_CONFIG`: Path to blueprint configuration file
- `IAAC_PREFERRED_SCENARIO`: Preferred deployment scenario

### 4. Backward Compatibility
All methods fall back to the constants package if blueprint configuration is not available, ensuring existing code continues to work.

## Usage Examples

### Basic Usage
```go
cfg, err := LoadConfig("")
if err != nil {
    log.Fatal(err)
}

// Get scenario path
path, err := cfg.GetScenarioPath("minimal")

// Validate scenario
err = cfg.ValidateScenario("full-stack")

// List scenarios
scenarios, err := cfg.ListAvailableScenarios()
```

### Custom Blueprint Config
```go
cfg.App.BlueprintConfig = "/path/to/custom/blueprint.yaml"
manager, err := cfg.GetBlueprintManager()
```

## Benefits

1. **Dynamic Configuration**: Paths and scenarios can be configured via blueprint YAML files
2. **Flexibility**: Users can provide custom blueprint configurations
3. **Backward Compatibility**: Falls back to hardcoded constants when blueprint config is not available
4. **Consistency**: Uses the same blueprint system as other parts of the codebase
5. **Extensibility**: Easy to add new scenarios and modules without code changes

## Testing
All existing tests pass, and new tests have been added to verify blueprint integration functionality.