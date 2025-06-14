# Agent Development Guide

This repository is transitioning from the existing Python implementation to a high-performance Go
version. Only OpenAI and PostgreSQL using the `pgvector` extension need to be supported in the
initial migration.

## Migration Plan
1. **Assess Current Code** – Review the Python modules in `gptcache` and `gptcache_server` to
   understand the caching API, data manager interfaces and server behaviour.
2. **Create Go Module** – Add a new directory `go/` at the repository root and initialize it with
   `go mod init`. Organize packages as `core`, `storage`, `openai`, and `server`.
3. **Core Cache Logic** – Implement a `Cache` struct that mirrors the behaviour of the Python
   `Cache` class (init, get, set). Use goroutines and sync primitives to maximise throughput.
4. **PostgreSQL Storage** – Provide a storage layer using `database/sql` with connection pooling
   and prepared statements. Use the `pgvector` extension to store and index embeddings for
   similarity search alongside prompts and responses.
5. **OpenAI Integration** – Implement a minimal client wrapper for the OpenAI API. Only the
   endpoints required by the current project need to be exposed.
6. **HTTP Server** – Implement a REST (or gRPC) server exposing cache operations. The API should be
   compatible with the existing Python server so other languages can talk to it.
7. **Documentation** – Document build steps, running the server and examples in the README. Update
   docs when new Go modules appear.

## Contribution Guidelines
- Place all Go code under the `go/` directory.
- Format Go files with `gofmt -w` and run `go vet ./...` and `go test ./...` before committing.
- Provide clear commit messages referencing the module or feature being ported.

Follow this plan for future work on the Go migration.
# AGENTS.md - Go Style Guide for AI Agents Development

*Comprehensive adaptation of Google's Go Style Guide for AI agents and intelligent systems development*

## Table of Contents

1. [Overview and Principles](#overview-and-principles)
2. [Core Style Guidelines](#core-style-guidelines)
3. [Naming Conventions](#naming-conventions)
4. [Code Organization](#code-organization)
5. [Error Handling](#error-handling)
6. [Concurrency and Goroutines](#concurrency-and-goroutines)
7. [Testing Best Practices](#testing-best-practices)
8. [Documentation Standards](#documentation-standards)
9. [Performance Considerations](#performance-considerations)
10. [Agent-Specific Patterns](#agent-specific-patterns)
11. [OpenTelemetry Context Propagation](#opentelemetry-context-propagation)

---

## Overview and Principles

### Foundation of Readable Code

**Clarity is the primary goal.** Code should be clear to the reader, not just the author. This is especially critical for AI agents systems where complex behaviors emerge from simple rules.

**Key Attributes of Readable Code (in order of importance):**

1. **Clarity** - What the code is doing should be obvious
2. **Simplicity** - Use the simplest approach that accomplishes the goal
3. **Concision** - High signal-to-noise ratio in code
4. **Maintainability** - Easy to modify and extend
5. **Consistency** - Follows established patterns throughout the codebase

### Guiding Principles for AI Agents

```go
// Good: Clear intent for agent behavior
type Agent struct {
    brain   *NeuralNetwork
    memory  *MemorySystem
    sensors []Sensor
}

func (a *Agent) Perceive(environment Environment) Perception {
    var signals []Signal
    for _, sensor := range a.sensors {
        signal := sensor.Read(environment)
        signals = append(signals, signal)
    }
    return a.brain.Process(signals)
}

// Bad: Unclear purpose and mixed responsibilities
type Thing struct {
    stuff   interface{}
    things  []interface{}
    data    map[string]interface{}
}

func (t *Thing) DoStuff(input interface{}) interface{} {
    // What does this actually do?
    return t.stuff
}
```

---

## Core Style Guidelines

### Formatting

**All Go source files must conform to `gofmt` output.** This is non-negotiable.

```bash
# Format all Go files
gofmt -w .

# Check formatting
gofmt -d .
```

### Mixed Caps Naming

Use `MixedCaps` or `mixedCaps` (camel case) rather than underscores:

```go
// Good
const MaxRetries = 3
const maxBufferSize = 1024
type NeuralNetwork struct{}
func (nn *NeuralNetwork) ForwardPass() {}

// Bad
const MAX_RETRIES = 3
const max_buffer_size = 1024
type Neural_Network struct{}
func (nn *Neural_Network) forward_pass() {}
```

### Line Length

**No fixed line length limit.** If a line feels too long, prefer refactoring over breaking it. Focus on clarity over arbitrary length constraints.

```go
// Good: Keep related logic together
agent := NewAgent(WithBrain(brain), WithMemory(memory), WithSensors(sensors))

// Bad: Arbitrary line breaks that hurt readability
agent := NewAgent(WithBrain(brain),
    WithMemory(memory), WithSensors(sensors))
```

---

## Naming Conventions

### Packages

Package names should be:
- **Short and concise**
- **All lowercase**
- **Single words when possible**
- **Related to what they provide**

```go
// Good
package agent
package brain
package memory
package sensor

// Bad
package agentPackage
package neural_network
package utilsAndHelpers
```

### Variables and Functions

#### Variable Names

The length of a name should be proportional to its scope:

```go
// Good: Short names in small scopes
for i, neuron := range layer.neurons {
    neuron.Activate(inputs[i])
}

// Good: Descriptive names in larger scopes
func TrainNeuralNetwork(trainingData Dataset, config TrainingConfig) (*NeuralNetwork, error) {
    neuralNetwork := NewNeuralNetwork(config.Architecture)
    optimizationAlgorithm := NewOptimizer(config.LearningRate)
    // ... training logic
}
```

#### Function Names

Functions that **return something** use noun-like names:
```go
func (a *Agent) CurrentState() State
func (nn *NeuralNetwork) Prediction() []float64
func (m *Memory) RecentExperiences() []Experience
```

Functions that **do something** use verb-like names:
```go
func (a *Agent) Act(environment Environment)
func (nn *NeuralNetwork) Train(data TrainingSet)
func (m *Memory) Store(experience Experience)
```

### Constants

Use `MixedCaps` for constants:

```go
// Good
const MaxIterations = 1000
const DefaultLearningRate = 0.01
const (
    StatusIdle = iota
    StatusTraining
    StatusInference
)

// Bad
const MAX_ITERATIONS = 1000
const kDefaultLearningRate = 0.01
```

### Receivers

Receiver names should be:
- **Short (1-2 characters)**
- **Consistent throughout the type**
- **Abbreviations of the type name**

```go
// Good
func (a *Agent) Think() Decision { }
func (a *Agent) Act(decision Decision) { }

func (nn *NeuralNetwork) Forward(input []float64) []float64 { }
func (nn *NeuralNetwork) Backward(gradient []float64) { }

// Bad
func (agent *Agent) Think() Decision { }
func (this *Agent) Act(decision Decision) { }
func (self *NeuralNetwork) Forward(input []float64) []float64 { }
```

---

## Code Organization

### Package Structure

Organize code into focused packages:

```
agent/
├── brain/           # Neural networks, decision making
│   ├── network.go
│   ├── layer.go
│   └── activation.go
├── memory/          # Experience storage and retrieval
│   ├── buffer.go
│   ├── experience.go
│   └── replay.go
├── sensor/          # Environment perception
│   ├── vision.go
│   ├── audio.go
│   └── sensor.go
├── environment/     # World simulation
│   ├── world.go
│   ├── physics.go
│   └── renderer.go
└── agent.go        # Main agent coordination
```

### Import Organization

Group imports into distinct blocks:

```go
import (
    // Standard library
    "context"
    "fmt"
    "math"
    "time"

    // Third-party packages
    "github.com/gorilla/websocket"
    "golang.org/x/sync/errgroup"

    // Local packages
    "myproject/agent/brain"
    "myproject/agent/memory"
    "myproject/agent/sensor"
)
```

### File Organization

Keep files focused and reasonably sized:

```go
// agent.go - Main agent coordination
type Agent struct {
    brain  *brain.Network
    memory *memory.Buffer
    sensors []sensor.Sensor
}

// brain/network.go - Neural network implementation
type Network struct {
    layers []Layer
    weights [][]float64
}

// memory/buffer.go - Experience buffer
type Buffer struct {
    experiences []Experience
    capacity    int
}
```

---

## Error Handling

### Error Return Values

Always handle errors explicitly. Use `error` as the last return value:

```go
// Good
func (a *Agent) LoadModel(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return fmt.Errorf("failed to read model file: %w", err)
    }
    
    if err := a.brain.LoadWeights(data); err != nil {
        return fmt.Errorf("failed to load weights: %w", err)
    }
    
    return nil
}

// Bad
func (a *Agent) LoadModel(path string) {
    data, _ := os.ReadFile(path) // Ignoring errors
    a.brain.LoadWeights(data)    // No error handling
}
```

### Error Handling Patterns

Handle errors early and return:

```go
// Good
func (a *Agent) ProcessInput(input []float64) ([]float64, error) {
    if len(input) == 0 {
        return nil, fmt.Errorf("input cannot be empty")
    }
    
    normalized, err := a.normalizeInput(input)
    if err != nil {
        return nil, fmt.Errorf("normalization failed: %w", err)
    }
    
    output, err := a.brain.Forward(normalized)
    if err != nil {
        return nil, fmt.Errorf("forward pass failed: %w", err)
    }
    
    return output, nil
}

// Bad
func (a *Agent) ProcessInput(input []float64) ([]float64, error) {
    if len(input) == 0 {
        return nil, fmt.Errorf("input cannot be empty")
    } else {
        normalized, err := a.normalizeInput(input)
        if err != nil {
            return nil, fmt.Errorf("normalization failed: %w", err)
        } else {
            output, err := a.brain.Forward(normalized)
            if err != nil {
                return nil, fmt.Errorf("forward pass failed: %w", err)
            } else {
                return output, nil
            }
        }
    }
}
```

### Structured Errors

Create structured errors for better error handling:

```go
// Good
type TrainingError struct {
    Epoch int
    Loss  float64
    Cause error
}

func (e *TrainingError) Error() string {
    return fmt.Sprintf("training failed at epoch %d (loss: %.4f): %v", 
        e.Epoch, e.Loss, e.Cause)
}

func (nn *NeuralNetwork) Train(data TrainingSet) error {
    for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
        loss, err := nn.trainEpoch(data)
        if err != nil {
            return &TrainingError{
                Epoch: epoch,
                Loss:  loss,
                Cause: err,
            }
        }
    }
    return nil
}
```

---

## Concurrency and Goroutines

### Goroutine Lifecycle Management

Make goroutine lifetimes obvious and manage them explicitly:

```go
// Good
func (a *Agent) Run(ctx context.Context) error {
    var wg sync.WaitGroup
    
    // Start sensor goroutines
    for _, sensor := range a.sensors {
        wg.Add(1)
        go func(s sensor.Sensor) {
            defer wg.Done()
            s.Run(ctx) // Sensor respects context cancellation
        }(sensor)
    }
    
    // Start main processing loop
    wg.Add(1)
    go func() {
        defer wg.Done()
        a.processLoop(ctx)
    }()
    
    wg.Wait() // Wait for all goroutines to finish
    return nil
}

// Bad
func (a *Agent) Run() {
    // Goroutines with no clear lifetime management
    for _, sensor := range a.sensors {
        go sensor.Run() // When do these stop?
    }
    go a.processLoop() // No way to stop this
}
```

### Channel Direction

Specify channel directions to make data flow clear:

```go
// Good
func (a *Agent) StartSensorPipeline(ctx context.Context) {
    sensorData := make(chan SensorReading, 100)
    processedData := make(chan ProcessedReading, 100)
    
    go a.collectSensorData(ctx, sensorData) // send-only
    go a.processSensorData(ctx, sensorData, processedData) // receive from first, send to second
    go a.consumeProcessedData(ctx, processedData) // receive-only
}

func (a *Agent) collectSensorData(ctx context.Context, output chan<- SensorReading) {
    defer close(output)
    // ... implementation
}

func (a *Agent) processSensorData(ctx context.Context, input <-chan SensorReading, output chan<- ProcessedReading) {
    defer close(output)
    // ... implementation
}

func (a *Agent) consumeProcessedData(ctx context.Context, input <-chan ProcessedReading) {
    // ... implementation
}
```

### Synchronous vs Asynchronous

Prefer synchronous functions for clarity:

```go
// Good: Synchronous with clear error handling
func (nn *NeuralNetwork) Train(data TrainingSet) error {
    for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
        if err := nn.trainEpoch(data); err != nil {
            return fmt.Errorf("epoch %d failed: %w", epoch, err)
        }
    }
    return nil
}

// Usage: caller controls concurrency if needed
go func() {
    if err := neuralNet.Train(trainingData); err != nil {
        log.Printf("Training failed: %v", err)
    }
}()

// Bad: Hidden concurrency makes error handling difficult
func (nn *NeuralNetwork) TrainAsync(data TrainingSet, callback func(error)) {
    go func() {
        // How do we handle errors here?
        for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
            nn.trainEpoch(data) // Errors lost
        }
        callback(nil)
    }()
}
```

---

## Testing Best Practices

### Table-Driven Tests

Use table-driven tests for comprehensive coverage:

```go
func TestNeuralNetwork_Forward(t *testing.T) {
    tests := []struct {
        name     string
        input    []float64
        weights  [][]float64
        expected []float64
        wantErr  bool
    }{
        {
            name:     "simple_forward_pass",
            input:    []float64{1.0, 0.5},
            weights:  [][]float64{{0.2, 0.8}, {0.4, 0.6}},
            expected: []float64{0.6, 0.8},
            wantErr:  false,
        },
        {
            name:     "empty_input",
            input:    []float64{},
            weights:  [][]float64{{0.2, 0.8}},
            expected: nil,
            wantErr:  true,
        },
        {
            name:     "mismatched_dimensions",
            input:    []float64{1.0},
            weights:  [][]float64{{0.2, 0.8, 0.3}},
            expected: nil,
            wantErr:  true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            nn := NewNeuralNetwork(len(tt.weights), len(tt.weights[0]))
            nn.SetWeights(tt.weights)
            
            got, err := nn.Forward(tt.input)
            if (err != nil) != tt.wantErr {
                t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            
            if !tt.wantErr && !reflect.DeepEqual(got, tt.expected) {
                t.Errorf("Forward() = %v, want %v", got, tt.expected)
            }
        })
    }
}
```

### Test Helpers

Create focused test helpers that set up test data:

```go
// Good: Focused helper for creating test agents
func newTestAgent(t *testing.T, options ...AgentOption) *Agent {
    t.Helper()
    
    config := DefaultConfig()
    for _, opt := range options {
        opt(&config)
    }
    
    agent, err := NewAgent(config)
    if err != nil {
        t.Fatalf("failed to create test agent: %v", err)
    }
    
    return agent
}

// Usage in tests
func TestAgent_Learn(t *testing.T) {
    agent := newTestAgent(t, 
        WithBrainSize(10),
        WithMemoryCapacity(1000),
    )
    
    experience := Experience{
        State:  []float64{1, 0, 1},
        Action: 2,
        Reward: 1.0,
    }
    
    err := agent.Learn(experience)
    if err != nil {
        t.Fatalf("Learn() failed: %v", err)
    }
}
```

### Mock Interfaces

Use interfaces for testability:

```go
// Good: Define interfaces for external dependencies
type Environment interface {
    GetState() State
    ApplyAction(Action) (State, Reward, bool)
    Reset() State
}

type MockEnvironment struct {
    states  []State
    rewards []Reward
    index   int
}

func (m *MockEnvironment) GetState() State {
    if m.index >= len(m.states) {
        return State{}
    }
    return m.states[m.index]
}

func (m *MockEnvironment) ApplyAction(action Action) (State, Reward, bool) {
    if m.index >= len(m.states)-1 {
        return State{}, 0, true // Episode done
    }
    m.index++
    return m.states[m.index], m.rewards[m.index], false
}

func TestAgent_Training(t *testing.T) {
    env := &MockEnvironment{
        states:  []State{{1, 0}, {0, 1}, {1, 1}},
        rewards: []Reward{0, 1, 5},
    }
    
    agent := newTestAgent(t)
    episode := agent.RunEpisode(env)
    
    if episode.TotalReward != 6 {
        t.Errorf("Expected total reward 6, got %v", episode.TotalReward)
    }
}
```

---

## Documentation Standards

### Package Documentation

Every package should have comprehensive documentation:

```go
// Package agent provides core functionality for autonomous agents.
//
// This package implements reinforcement learning agents that can learn
// to navigate environments through trial and error. The main components are:
//
//   - Agent: The main controller that coordinates perception, decision-making, and learning
//   - Brain: Neural network for decision making
//   - Memory: Experience replay buffer for learning
//   - Sensor: Interface for environment perception
//
// Basic usage:
//
//   config := agent.DefaultConfig()
//   config.LearningRate = 0.001
//   
//   a, err := agent.New(config)
//   if err != nil {
//       log.Fatal(err)
//   }
//   
//   for episode := 0; episode < 1000; episode++ {
//       a.RunEpisode(environment)
//   }
package agent
```

### Function Documentation

Document all exported functions with clear examples:

```go
// Train trains the neural network using the provided training data.
//
// The training process uses backpropagation with the configured optimizer
// and runs for the specified number of epochs. Training can be stopped
// early if the loss converges below the threshold.
//
// Returns an error if the training data is invalid or if training fails
// to converge within the maximum number of epochs.
//
// Example:
//   
//   nn := brain.NewNeuralNetwork([]int{784, 128, 10})
//   trainingData := loadMNISTData()
//   
//   if err := nn.Train(trainingData); err != nil {
//       log.Fatalf("Training failed: %v", err)
//   }
func (nn *NeuralNetwork) Train(data TrainingSet) error {
    if len(data.Inputs) == 0 {
        return fmt.Errorf("training data cannot be empty")
    }
    
    for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
        if err := nn.trainEpoch(data); err != nil {
            return fmt.Errorf("epoch %d failed: %w", epoch, err)
        }
    }
    
    return nil
}
```

### Type Documentation

Document important types and their usage:

```go
// Agent represents an autonomous learning agent.
//
// An Agent perceives its environment through sensors, makes decisions
// using its brain (neural network), and learns from experiences stored
// in memory. Agents can be trained using reinforcement learning algorithms.
//
// The Agent maintains internal state including:
//   - Current policy (decision-making strategy)
//   - Experience replay buffer
//   - Learning statistics
//
// Example:
//   
//   agent := &Agent{
//       Brain:   brain.NewNetwork([]int{4, 64, 2}),
//       Memory:  memory.NewBuffer(10000),
//       Sensors: []Sensor{vision.New(), audio.New()},
//   }
//   
//   agent.Train(environment, 1000)
type Agent struct {
    // Brain handles decision making and learning
    Brain *brain.Network
    
    // Memory stores experiences for replay learning
    Memory *memory.Buffer
    
    // Sensors provide environment perception
    Sensors []Sensor
    
    // config holds training and runtime configuration
    config Config
}
```

---

## Performance Considerations

### Memory Management

#### Slice and Map Initialization

Specify capacity when size is known:

```go
// Good: Pre-allocate with known capacity
func (b *Buffer) AddExperiences(experiences []Experience) {
    if cap(b.data) < len(b.data)+len(experiences) {
        newCap := max(cap(b.data)*2, len(b.data)+len(experiences))
        newData := make([]Experience, len(b.data), newCap)
        copy(newData, b.data)
        b.data = newData
    }
    b.data = append(b.data, experiences...)
}

// Good: Initialize map with expected size
func NewNeuralNetwork(layerSizes []int) *NeuralNetwork {
    weights := make(map[string][]float64, len(layerSizes))
    for i, size := range layerSizes {
        weights[fmt.Sprintf("layer_%d", i)] = make([]float64, size)
    }
    return &NeuralNetwork{weights: weights}
}
```

#### Avoid Copying Large Structs

Use pointers for large structs:

```go
// Good: Use pointers for large structs
type Experience struct {
    State      []float64 // potentially large
    Action     int
    Reward     float64
    NextState  []float64 // potentially large
    Done       bool
}

func (m *Memory) Store(exp *Experience) {
    m.buffer = append(m.buffer, exp)
}

// Bad: Copying large structs
func (m *Memory) Store(exp Experience) { // copies entire struct
    m.buffer = append(m.buffer, &exp)
}
```

### String Building

Use `strings.Builder` for efficient string concatenation:

```go
// Good: Use strings.Builder for multiple concatenations
func (a *Agent) LogPerformance(episodes []Episode) string {
    var b strings.Builder
    b.WriteString("Performance Report:\n")
    
    for i, ep := range episodes {
        fmt.Fprintf(&b, "Episode %d: Reward=%.2f, Steps=%d\n", 
            i, ep.TotalReward, ep.StepCount)
    }
    
    return b.String()
}

// Bad: Multiple string concatenations
func (a *Agent) LogPerformance(episodes []Episode) string {
    result := "Performance Report:\n"
    for i, ep := range episodes {
        result += fmt.Sprintf("Episode %d: Reward=%.2f, Steps=%d\n", 
            i, ep.TotalReward, ep.StepCount)
    }
    return result
}
```

---

## Agent-Specific Patterns

### Configuration Patterns

Use the functional options pattern for complex configuration:

```go
// Good: Functional options for agent configuration
type AgentConfig struct {
    LearningRate    float64
    MemoryCapacity  int
    NetworkArchitecture []int
    ExplorationRate float64
}

type AgentOption func(*AgentConfig)

func WithLearningRate(rate float64) AgentOption {
    return func(c *AgentConfig) {
        c.LearningRate = rate
    }
}

func WithMemoryCapacity(capacity int) AgentOption {
    return func(c *AgentConfig) {
        c.MemoryCapacity = capacity
    }
}

func WithNetworkArchitecture(layers []int) AgentOption {
    return func(c *AgentConfig) {
        c.NetworkArchitecture = layers
    }
}

func NewAgent(options ...AgentOption) (*Agent, error) {
    config := AgentConfig{
        LearningRate:    0.001,
        MemoryCapacity:  10000,
        NetworkArchitecture: []int{4, 64, 64, 2},
        ExplorationRate: 0.1,
    }
    
    for _, opt := range options {
        opt(&config)
    }
    
    return &Agent{
        brain:  brain.NewNetwork(config.NetworkArchitecture),
        memory: memory.NewBuffer(config.MemoryCapacity),
        config: config,
    }, nil
}

// Usage
agent, err := NewAgent(
    WithLearningRate(0.01),
    WithMemoryCapacity(50000),
    WithNetworkArchitecture([]int{8, 128, 128, 4}),
)
```

### State Management

Implement clear state transitions:

```go
// Good: Clear state management
type AgentState int

const (
    StateIdle AgentState = iota
    StateTraining
    StateInference
    StateUpdating
)

type Agent struct {
    state      AgentState
    stateMutex sync.RWMutex
    // ... other fields
}

func (a *Agent) SetState(newState AgentState) {
    a.stateMutex.Lock()
    defer a.stateMutex.Unlock()
    
    // Validate state transition
    if !a.isValidTransition(a.state, newState) {
        log.Printf("Invalid state transition from %v to %v", a.state, newState)
        return
    }
    
    oldState := a.state
    a.state = newState
    log.Printf("Agent state changed: %v -> %v", oldState, newState)
}

func (a *Agent) GetState() AgentState {
    a.stateMutex.RLock()
    defer a.stateMutex.RUnlock()
    return a.state
}

func (a *Agent) isValidTransition(from, to AgentState) bool {
    validTransitions := map[AgentState][]AgentState{
        StateIdle:      {StateTraining, StateInference},
        StateTraining:  {StateIdle, StateUpdating},
        StateInference: {StateIdle, StateUpdating},
        StateUpdating:  {StateIdle, StateTraining, StateInference},
    }
    
    allowed := validTransitions[from]
    for _, valid := range allowed {
        if valid == to {
            return true
        }
    }
    return false
}
```

### Learning Algorithms

Structure learning algorithms clearly:

```go
// Good: Clear learning algorithm structure
type LearningAlgorithm interface {
    Learn(experience *Experience) error
    UpdateTarget() error
    GetPolicy() Policy
}

type DQNAlgorithm struct {
    mainNetwork   *brain.Network
    targetNetwork *brain.Network
    memory        *memory.Buffer
    config        DQNConfig
}

func (dqn *DQNAlgorithm) Learn(experience *Experience) error {
    // Store experience
    dqn.memory.Store(experience)
    
    // Check if we have enough experiences to learn
    if dqn.memory.Size() < dqn.config.MinMemorySize {
        return nil
    }
    
    // Sample batch from memory
    batch, err := dqn.memory.SampleBatch(dqn.config.BatchSize)
    if err != nil {
        return fmt.Errorf("failed to sample batch: %w", err)
    }
    
    // Compute targets using target network
    targets, err := dqn.computeTargets(batch)
    if err != nil {
        return fmt.Errorf("failed to compute targets: %w", err)
    }
    
    // Train main network
    if err := dqn.mainNetwork.TrainBatch(batch, targets); err != nil {
        return fmt.Errorf("training failed: %w", err)
    }
    
    return nil
}

func (dqn *DQNAlgorithm) computeTargets(batch []*Experience) ([][]float64, error) {
    targets := make([][]float64, len(batch))
    
    for i, exp := range batch {
        if exp.Done {
            targets[i] = []float64{exp.Reward}
        } else {
            nextQ, err := dqn.targetNetwork.Forward(exp.NextState)
            if err != nil {
                return nil, fmt.Errorf("target network forward pass failed: %w", err)
            }
            maxQ := maxValue(nextQ)
            targets[i] = []float64{exp.Reward + dqn.config.Gamma*maxQ}
        }
    }
    
    return targets, nil
}
```

### Environment Interaction

Define clear interfaces for environment interaction:

```go
// Good: Clear environment interface
type Environment interface {
    // Reset resets the environment to initial state
    Reset() (State, error)
    
    // Step executes an action and returns the result
    Step(action Action) (State, Reward, bool, error)
    
    // GetActionSpace returns the valid actions
    GetActionSpace() ActionSpace
    
    // GetStateSpace returns the state space description
    GetStateSpace() StateSpace
    
    // Render displays the current environment state
    Render() error
    
    // Close cleans up environment resources
    Close() error
}

 type ActionSpace struct {
    Type string // "discrete" or "continuous"
    Size int    // number of discrete actions or dimensions
    Low  []float64 // minimum values for continuous actions
    High []float64 // maximum values for continuous actions
 }

 type StateSpace struct {
    Shape []int     // dimensions of state
    Low   []float64 // minimum values
    High  []float64 // maximum values
 }

 // Usage in agent training loop
 func (a *Agent) RunEpisode(env Environment) (*Episode, error) {
    state, err := env.Reset()
    if err != nil {
        return nil, fmt.Errorf("failed to reset environment: %w", err)
    }
    
    episode := &Episode{
        Steps: make([]Step, 0),
    }
    
    for step := 0; step < a.config.MaxSteps; step++ {
        action, err := a.SelectAction(state)
        if err != nil {
            return nil, fmt.Errorf("action selection failed: %w", err)
        }
        
        nextState, reward, done, err := env.Step(action)
        if err != nil {
            return nil, fmt.Errorf("environment step failed: %w", err)
        }
        
        experience := &Experience{
            State:     state,
            Action:    action,
            Reward:    reward,
            NextState: nextState,
            Done:      done,
        }
        
        if err := a.Learn(experience); err != nil {
            log.Printf("Learning failed: %v", err)
        }
        
        episode.Steps = append(episode.Steps, Step{
            State:  state,
            Action: action,
            Reward: reward,
        })
        episode.TotalReward += reward
        
        if done {
            break
        }
        
        state = nextState
    }
    
    return episode, nil
 }
```

---

## OpenTelemetry Context Propagation

Observability is crucial for AI agents systems. OpenTelemetry provides standardized tracing, metrics, and logging that help understand agent behavior, performance bottlenecks, and system interactions.

### Context Propagation Fundamentals

Always propagate context through the entire call chain to maintain trace continuity:

```go
import (
    "context"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/trace"
)

// Good: Context propagation through agent operations
func (a *Agent) RunEpisode(ctx context.Context, env Environment) (*Episode, error) {
    tracer := otel.Tracer("agent")
    ctx, span := tracer.Start(ctx, "agent.run_episode",
        trace.WithAttributes(
            attribute.String("agent.id", a.ID),
            attribute.String("environment.type", env.Type()),
        ),
    )
    defer span.End()
    
    // Reset environment with context
    state, err := a.resetEnvironment(ctx, env)
    if err != nil {
        span.SetStatus(codes.Error, "failed to reset environment")
        span.RecordError(err)
        return nil, err
    }
    
    episode := &Episode{ID: generateEpisodeID()}
    span.SetAttributes(attribute.String("episode.id", episode.ID))
    
    for step := 0; step < a.config.MaxSteps; step++ {
        // Each step gets its own span with context
        stepCtx, stepSpan := tracer.Start(ctx, "agent.step",
            trace.WithAttributes(
                attribute.Int("step.number", step),
                attribute.String("episode.id", episode.ID),
            ),
        )
        
        action, err := a.selectAction(stepCtx, state)
        if err != nil {
            stepSpan.SetStatus(codes.Error, "action selection failed")
            stepSpan.RecordError(err)
            stepSpan.End()
            continue
        }
        
        nextState, reward, done, err := a.executeAction(stepCtx, env, action)
        stepSpan.SetAttributes(
            attribute.Int("action.value", int(action)),
            attribute.Float64("reward.value", reward),
            attribute.Bool("episode.done", done),
        )
        
        if err != nil {
            stepSpan.SetStatus(codes.Error, "action execution failed")
            stepSpan.RecordError(err)
        }
        
        stepSpan.End()
        
        // Learn from experience with context
        if err := a.learn(ctx, &Experience{
            State: state, Action: action, Reward: reward,
            NextState: nextState, Done: done,
        }); err != nil {
            span.AddEvent("learning_failed", trace.WithAttributes(
                attribute.String("error", err.Error()),
            ))
        }
        
        episode.TotalReward += reward
        if done {
            break
        }
        state = nextState
    }
    
    span.SetAttributes(
        attribute.Float64("episode.total_reward", episode.TotalReward),
        attribute.Int("episode.steps", len(episode.Steps)),
    )
    
    return episode, nil
}

// Bad: No context propagation
func (a *Agent) RunEpisode(env Environment) (*Episode, error) {
    // No tracing context - loses observability
    state, err := env.Reset()
    // ... rest of implementation without context
}
```

### Tracer Initialization and Configuration

Set up OpenTelemetry properly at application startup:

```go
// Good: Proper tracer initialization
package main

import (
    "context"
    "log"
    "os"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
)

func initTracer() func() {
    // Create Jaeger exporter
    exp, err := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint(os.Getenv("JAEGER_ENDPOINT")),
    ))
    if err != nil {
        log.Fatalf("Failed to create Jaeger exporter: %v", err)
    }
    
    // Create resource with service information
    res, err := resource.New(context.Background(),
        resource.WithAttributes(
            semconv.ServiceNameKey.String("ai-agent-system"),
            semconv.ServiceVersionKey.String("v1.0.0"),
            semconv.DeploymentEnvironmentKey.String(os.Getenv("ENVIRONMENT")),
        ),
    )
    if err != nil {
        log.Fatalf("Failed to create resource: %v", err)
    }
    
    // Create trace provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exp),
        trace.WithResource(res),
        trace.WithSampler(trace.AlwaysSample()), // Adjust sampling for production
    )
    
    // Set global tracer provider
    otel.SetTracerProvider(tp)
    
    // Set global propagator for context propagation
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))
    
    return func() {
        if err := tp.Shutdown(context.Background()); err != nil {
            log.Printf("Failed to shutdown tracer: %v", err)
        }
    }
}

func main() {
    cleanup := initTracer()
    defer cleanup()
    
    agent := NewAgent()
    ctx := context.Background()
    
    // Start training with root context
    if err := agent.Train(ctx, environment, 1000); err != nil {
        log.Fatalf("Training failed: %v", err)
    }
}
```

### Neural Network Training Observability

Instrument training loops with detailed metrics and traces:

```go
// Good: Comprehensive training observability
func (nn *NeuralNetwork) Train(ctx context.Context, data TrainingSet) error {
    tracer := otel.Tracer("neural-network")
    meter := otel.Meter("neural-network")
    
    // Create metrics
    lossCounter, _ := meter.Float64Counter("training.loss")
    epochCounter, _ := meter.Int64Counter("training.epochs")
    
    ctx, span := tracer.Start(ctx, "neural_network.train",
        trace.WithAttributes(
            attribute.Int("training.batch_size", data.BatchSize),
            attribute.Int("training.epochs", nn.config.MaxEpochs),
            attribute.Float64("training.learning_rate", nn.config.LearningRate),
            attribute.IntSlice("network.architecture", nn.config.Architecture),
        ),
    )
    defer span.End()
    
    for epoch := 0; epoch < nn.config.MaxEpochs; epoch++ {
        epochCtx, epochSpan := tracer.Start(ctx, "neural_network.train_epoch",
            trace.WithAttributes(attribute.Int("epoch.number", epoch)),
        )
        
        loss, accuracy, err := nn.trainEpoch(epochCtx, data)
        if err != nil {
            epochSpan.SetStatus(codes.Error, "epoch training failed")
            epochSpan.RecordError(err)
            epochSpan.End()
            return fmt.Errorf("epoch %d failed: %w", epoch, err)
        }
        
        // Record metrics
        lossCounter.Add(ctx, loss, attribute.Int("epoch", epoch))
        epochCounter.Add(ctx, 1)
        
        // Add span attributes
        epochSpan.SetAttributes(
            attribute.Float64("epoch.loss", loss),
            attribute.Float64("epoch.accuracy", accuracy),
        )
        
        // Add events for significant milestones
        if accuracy > nn.bestAccuracy {
            nn.bestAccuracy = accuracy
            epochSpan.AddEvent("new_best_accuracy", trace.WithAttributes(
                attribute.Float64("best_accuracy", accuracy),
            ))
        }
        
        epochSpan.End()
        
        // Early stopping with trace event
        if loss < nn.config.ConvergenceThreshold {
            span.AddEvent("early_convergence", trace.WithAttributes(
                attribute.Int("final_epoch", epoch),
                attribute.Float64("final_loss", loss),
            ))
            break
        }
    }
    
    span.SetAttributes(
        attribute.Float64("training.final_loss", nn.currentLoss),
        attribute.Float64("training.final_accuracy", nn.currentAccuracy),
        attribute.Bool("training.converged", nn.converged),
    )
    
    return nil
}

func (nn *NeuralNetwork) trainEpoch(ctx context.Context, data TrainingSet) (float64, float64, error) {
    tracer := otel.Tracer("neural-network")
    ctx, span := tracer.Start(ctx, "neural_network.train_epoch")
    defer span.End()
    
    var totalLoss float64
    var correct int
    
    for batchIdx, batch := range data.Batches {
        batchCtx, batchSpan := tracer.Start(ctx, "neural_network.train_batch",
            trace.WithAttributes(
                attribute.Int("batch.index", batchIdx),
                attribute.Int("batch.size", len(batch.Inputs)),
            ),
        )
        
        // Forward pass
        predictions, err := nn.forward(batchCtx, batch.Inputs)
        if err != nil {
            batchSpan.RecordError(err)
            batchSpan.End()
            return 0, 0, err
        }
        
        // Compute loss
        batchLoss := nn.computeLoss(predictions, batch.Targets)
        totalLoss += batchLoss
        
        // Backward pass
        if err := nn.backward(batchCtx, batchLoss); err != nil {
            batchSpan.RecordError(err)
            batchSpan.End()
            return 0, 0, err
        }
        
        // Count correct predictions
        batchCorrect := nn.countCorrect(predictions, batch.Targets)
        correct += batchCorrect
        
        batchSpan.SetAttributes(
            attribute.Float64("batch.loss", batchLoss),
            attribute.Int("batch.correct", batchCorrect),
            attribute.Float64("batch.accuracy", float64(batchCorrect)/float64(len(batch.Targets))),
        )
        batchSpan.End()
    }
    
    epochLoss := totalLoss / float64(len(data.Batches))
    epochAccuracy := float64(correct) / float64(data.TotalSamples)
    
    span.SetAttributes(
        attribute.Float64("epoch.avg_loss", epochLoss),
        attribute.Float64("epoch.accuracy", epochAccuracy),
        attribute.Int("epoch.total_correct", correct),
        attribute.Int("epoch.total_samples", data.TotalSamples),
    )
    
    return epochLoss, epochAccuracy, nil
}
```

### Memory System Instrumentation

Track memory usage and replay buffer operations:

```go
// Good: Memory system with observability
import (
    "context"
    "fmt"
    "math/rand"
    "sync"
    
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/codes"
    "go.opentelemetry.io/otel/metric"
    "go.opentelemetry.io/otel/trace"
)

type Memory struct {
    buffer     []Experience
    capacity   int
    writeIndex int
    size       int
    mutex      sync.RWMutex
    
    // Metrics
    storeCounter    metric.Int64Counter
    sampleCounter   metric.Int64Counter
}

func NewMemory(capacity int) *Memory {
    meter := otel.Meter("memory")
    
    storeCounter, _ := meter.Int64Counter("memory.store_operations")
    sampleCounter, _ := meter.Int64Counter("memory.sample_operations")
    
    m := &Memory{
        buffer:           make([]Experience, capacity),
        capacity:         capacity,
        storeCounter:     storeCounter,
        sampleCounter:    sampleCounter,
    }
    
    return m
}

func (m *Memory) Store(ctx context.Context, exp *Experience) error {
    tracer := otel.Tracer("memory")
    ctx, span := tracer.Start(ctx, "memory.store",
        trace.WithAttributes(
            attribute.Float64("experience.reward", exp.Reward),
            attribute.Bool("experience.done", exp.Done),
        ),
    )
    defer span.End()
    
    m.mutex.Lock()
    defer m.mutex.Unlock()
    
    // Store experience
    m.buffer[m.writeIndex] = *exp
    m.writeIndex = (m.writeIndex + 1) % m.capacity
    
    if m.size < m.capacity {
        m.size++
    }
    
    // Update metrics
    m.storeCounter.Add(ctx, 1)
    utilization := float64(m.size) / float64(m.capacity)
    
    span.SetAttributes(
        attribute.Int("memory.size", m.size),
        attribute.Int("memory.write_index", m.writeIndex),
        attribute.Float64("memory.utilization", utilization),
    )
    
    return nil
}

func (m *Memory) SampleBatch(ctx context.Context, batchSize int) ([]*Experience, error) {
    tracer := otel.Tracer("memory")
    ctx, span := tracer.Start(ctx, "memory.sample_batch",
        trace.WithAttributes(
            attribute.Int("batch.requested_size", batchSize),
        ),
    )
    defer span.End()
    
    m.mutex.RLock()
    defer m.mutex.RUnlock()
    
    if m.size < batchSize {
        err := fmt.Errorf("insufficient experiences: have %d, need %d", m.size, batchSize)
        span.SetStatus(codes.Error, "insufficient experiences")
        span.RecordError(err)
        return nil, err
    }
    
    batch := make([]*Experience, batchSize)
    indices := make([]int, batchSize)
    
    // Sample random indices
    for i := 0; i < batchSize; i++ {
        indices[i] = rand.Intn(m.size)
        batch[i] = &m.buffer[indices[i]]
    }
    
    // Update metrics
    m.sampleCounter.Add(ctx, 1, attribute.Int("batch.size", batchSize))
    
    span.SetAttributes(
        attribute.Int("batch.actual_size", len(batch)),
        attribute.IntSlice("batch.indices", indices),
    )
    
    return batch, nil
}
```

### Distributed Agent System Tracing

For multi-agent systems, ensure trace context propagation across services:

```go
type AgentCoordinator struct {
    agents    []Agent
    messenger MessageBroker
}

func (ac *AgentCoordinator) CoordinateAgents(ctx context.Context, task Task) error {
    tracer := otel.Tracer("coordinator")
    ctx, span := tracer.Start(ctx, "coordinator.coordinate_agents",
        trace.WithAttributes(
            attribute.String("task.id", task.ID),
            attribute.String("task.type", task.Type),
            attribute.Int("agents.count", len(ac.agents)),
        ),
    )
    defer span.End()
    
    // Distribute task to agents
    var wg sync.WaitGroup
    results := make(chan AgentResult, len(ac.agents))
    
    for i, agent := range ac.agents {
        wg.Add(1)
        go func(agentID int, a Agent) {
            defer wg.Done()
            
            // Create child span for each agent
            agentCtx, agentSpan := tracer.Start(ctx, "coordinator.agent_execution",
                trace.WithAttributes(
                    attribute.Int("agent.id", agentID),
                    attribute.String("task.id", task.ID),
                ),
            )
            defer agentSpan.End()
            
            // Execute task with context propagation
            result, err := a.ExecuteTask(agentCtx, task)
            if err != nil {
                agentSpan.SetStatus(codes.Error, "agent execution failed")
                agentSpan.RecordError(err)
                return
            }
            
            agentSpan.SetAttributes(
                attribute.Float64("result.score", result.Score),
                attribute.String("result.status", result.Status),
            )
            
            results <- AgentResult{
                AgentID: agentID,
                Result:  result,
            }
        }(i, agent)
    }
    
    // Wait for all agents to complete
    go func() {
        wg.Wait()
        close(results)
    }()
    
    // Collect results
    var allResults []AgentResult
    for result := range results {
        allResults = append(allResults, result)
    }
    
    // Aggregate results with tracing
    finalResult, err := ac.aggregateResults(ctx, allResults)
    if err != nil {
        span.SetStatus(codes.Error, "result aggregation failed")
        span.RecordError(err)
        return err
    }
    
    span.SetAttributes(
        attribute.Int("results.count", len(allResults)),
        attribute.Float64("final_result.score", finalResult.Score),
    )
    
    return nil
}

func (a *Agent) ExecuteTask(ctx context.Context, task Task) (*TaskResult, error) {
    // Extract trace context from incoming request
    tracer := otel.Tracer("agent")
    ctx, span := tracer.Start(ctx, "agent.execute_task",
        trace.WithAttributes(
            attribute.String("agent.id", a.ID),
            attribute.String("task.id", task.ID),
        ),
    )
    defer span.End()
    
    // Perception phase
    perception, err := a.perceive(ctx, task.Environment)
    if err != nil {
        span.RecordError(err)
        return nil, err
    }
    
    // Decision phase
    decision, err := a.decide(ctx, perception)
    if err != nil {
        span.RecordError(err)
        return nil, err
    }
    
    // Action phase
    result, err := a.act(ctx, decision)
    if err != nil {
        span.RecordError(err)
        return nil, err
    }
    
    span.SetAttributes(
        attribute.String("perception.type", perception.Type),
        attribute.String("decision.action", decision.Action),
        attribute.Float64("result.confidence", result.Confidence),
    )
    
    return result, nil
}
```

### HTTP/gRPC Context Propagation

For agent services exposed via HTTP or gRPC:

```go
// Good: HTTP handler with context propagation
func (s *AgentService) TrainHandler(w http.ResponseWriter, r *http.Request) {
    // Extract trace context from HTTP headers
    ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
    
    tracer := otel.Tracer("agent-service")
    ctx, span := tracer.Start(ctx, "agent_service.train_handler",
        trace.WithAttributes(
            attribute.String("http.method", r.Method),
            attribute.String("http.url", r.URL.String()),
            attribute.String("http.user_agent", r.UserAgent()),
        ),
    )
    defer span.End()
    
    var req TrainRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        span.SetStatus(codes.Error, "invalid request")
        span.RecordError(err)
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }
    
    // Pass context to business logic
    result, err := s.agent.Train(ctx, req.Episodes, req.Environment)
    if err != nil {
        span.SetStatus(codes.Error, "training failed")
        span.RecordError(err)
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    span.SetAttributes(
        attribute.Int("training.episodes", req.Episodes),
        attribute.Float64("training.final_reward", result.FinalReward),
    )
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(result)
}

// gRPC service with context propagation
func (s *AgentGRPCService) Train(ctx context.Context, req *pb.TrainRequest) (*pb.TrainResponse, error) {
    tracer := otel.Tracer("agent-grpc-service")
    ctx, span := tracer.Start(ctx, "agent_grpc_service.train",
        trace.WithAttributes(
            attribute.String("grpc.service", "AgentService"),
            attribute.String("grpc.method", "Train"),
            attribute.Int("request.episodes", int(req.Episodes)),
        ),
    )
    defer span.End()
    
    // Business logic with context
    result, err := s.agent.Train(ctx, int(req.Episodes), req.Environment)
    if err != nil {
        span.SetStatus(codes.Error, "training failed")
        span.RecordError(err)
        return nil, status.Errorf(codes.Internal, "training failed: %v", err)
    }
    
    span.SetAttributes(
        attribute.Float64("result.final_reward", result.FinalReward),
        attribute.Bool("result.converged", result.Converged),
    )
    
    return &pb.TrainResponse{
        FinalReward: result.FinalReward,
        Converged:   result.Converged,
    }, nil
}
```

### Error Handling with Traces

Enrich error handling with trace information:

```go
// Good: Error handling with trace correlation
func (a *Agent) ProcessDecision(ctx context.Context, input DecisionInput) (*Decision, error) {
    tracer := otel.Tracer("agent")
    ctx, span := tracer.Start(ctx, "agent.process_decision")
    defer span.End()
    
    // Validate input
    if err := input.Validate(); err != nil {
        span.SetStatus(codes.Error, "invalid input")
        span.RecordError(err)
        span.SetAttributes(attribute.String("validation.error", err.Error()))
        return nil, fmt.Errorf("input validation failed: %w", err)
    }
    
    // Neural network inference
    prediction, err := a.brain.Predict(ctx, input.Features)
    if err != nil {
        span.SetStatus(codes.Error, "prediction failed")
        span.RecordError(err)
        span.SetAttributes(
            attribute.String("brain.error", err.Error()),
            attribute.Int("input.feature_count", len(input.Features)),
        )
        return nil, fmt.Errorf("brain prediction failed: %w", err)
    }
    
    // Decision logic
    decision, err := a.makeDecision(ctx, prediction, input.Context)
    if err != nil {
        span.SetStatus(codes.Error, "decision making failed")
        span.RecordError(err)
        return nil, fmt.Errorf("decision making failed: %w", err)
    }
    
    span.SetAttributes(
        attribute.String("decision.type", decision.Type),
        attribute.Float64("decision.confidence", decision.Confidence),
        attribute.Bool("decision.requires_action", decision.RequiresAction),
    )
    
    return decision, nil
}

// Custom error with trace information
type AgentError struct {
    Operation string
    TraceID   string
    SpanID    string
    Cause     error
}

func (e *AgentError) Error() string {
    return fmt.Sprintf("agent operation '%s' failed (trace: %s, span: %s): %v", 
        e.Operation, e.TraceID, e.SpanID, e.Cause)
}

func NewAgentError(ctx context.Context, operation string, cause error) *AgentError {
    spanCtx := trace.SpanContextFromContext(ctx)
    return &AgentError{
        Operation: operation,
        TraceID:   spanCtx.TraceID().String(),
        SpanID:    spanCtx.SpanID().String(),
        Cause:     cause,
    }
}
```

### Testing with OpenTelemetry

Test trace propagation and observability:

```go
// Good: Testing with trace verification
func TestAgent_TrainWithTracing(t *testing.T) {
    // Setup in-memory span exporter for testing
    exporter := tracetest.NewInMemoryExporter()
    tp := trace.NewTracerProvider(
        trace.WithSyncer(exporter),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("test-agent"),
        )),
    )
    otel.SetTracerProvider(tp)
    defer func() { _ = tp.Shutdown(context.Background()) }()
    
    agent := NewAgent()
    ctx := context.Background()
    
    // Execute training
    err := agent.Train(ctx, 10)
    require.NoError(t, err)
    
    // Verify traces were created
    spans := exporter.GetSpans()
    require.NotEmpty(t, spans, "Expected spans to be created")
    
    // Verify span hierarchy
    rootSpan := findSpanByName(spans, "agent.train")
    require.NotNil(t, rootSpan, "Expected root training span")
    
    episodeSpans := findSpansByName(spans, "agent.run_episode")
    assert.Equal(t, 10, len(episodeSpans), "Expected 10 episode spans")
    
    // Verify span attributes
    assert.Equal(t, "agent.train", rootSpan.Name)
    assert.Contains(t, rootSpan.Attributes, attribute.Int("training.episodes", 10))
    
    // Verify trace context propagation
    for _, episodeSpan := range episodeSpans {
        assert.Equal(t, rootSpan.SpanContext.TraceID(), 
            episodeSpan.SpanContext.TraceID(), 
            "Episode spans should have same trace ID as root")
    }
}

func TestAgent_ErrorTracing(t *testing.T) {
    exporter := tracetest.NewInMemoryExporter()
    tp := trace.NewTracerProvider(trace.WithSyncer(exporter))
    otel.SetTracerProvider(tp)
    defer func() { _ = tp.Shutdown(context.Background()) }()
    
    agent := NewAgent()
    ctx := context.Background()
    
    // Simulate error condition
    agent.brain = nil // This should cause an error
    
    err := agent.Train(ctx, 1)
    require.Error(t, err)
    
    // Verify error was recorded in trace
    spans := exporter.GetSpans()
    errorSpan := findSpanWithError(spans)
    require.NotNil(t, errorSpan, "Expected span with error status")
    
    assert.Equal(t, codes.Error, errorSpan.Status.Code)
    assert.Contains(t, errorSpan.Status.Description, "brain")
}

// Test helpers
func findSpanByName(spans []trace.ReadOnlySpan, name string) trace.ReadOnlySpan {
    for _, span := range spans {
        if span.Name() == name {
            return span
        }
    }
    return nil
}

func findSpansByName(spans []trace.ReadOnlySpan, name string) []trace.ReadOnlySpan {
    var result []trace.ReadOnlySpan
    for _, span := range spans {
        if span.Name() == name {
            result = append(result, span)
        }
    }
    return result
}

func findSpanWithError(spans []trace.ReadOnlySpan) trace.ReadOnlySpan {
    for _, span := range spans {
        if span.Status().Code == codes.Error {
            return span
        }
    }
    return nil
}
```

### Performance Considerations

Monitor the performance impact of observability:

```go
// Good: Conditional tracing for performance
type TracingConfig struct {
    Enabled       bool
    SampleRate    float64
    DetailedSpans bool
}

func (a *Agent) Train(ctx context.Context, episodes int) error {
    // Check if tracing is enabled
    if !a.config.Tracing.Enabled {
        return a.trainWithoutTracing(episodes)
    }
    
    tracer := otel.Tracer("agent")
    
    // Use sampling for high-frequency operations
    sampler := trace.TraceIDRatioBased(a.config.Tracing.SampleRate)
    if !sampler.ShouldSample(trace.SamplingParameters{
        ParentContext: ctx,
        TraceID:       trace.TraceIDFromContext(ctx),
    }).Decision.IsSampled() {
        return a.trainWithoutTracing(episodes)
    }
    
    ctx, span := tracer.Start(ctx, "agent.train")
    defer span.End()
    
    // Detailed spans only when configured
    if a.config.Tracing.DetailedSpans {
        return a.trainWithDetailedTracing(ctx, episodes)
    }
    
    return a.trainWithBasicTracing(ctx, episodes)
}

// Batch span creation for high-frequency operations
func (nn *NeuralNetwork) TrainBatch(ctx context.Context, batch TrainingBatch) error {
    tracer := otel.Tracer("neural-network")
    
    // Create single span for entire batch instead of per-sample spans
    ctx, span := tracer.Start(ctx, "neural_network.train_batch",
        trace.WithAttributes(
            attribute.Int("batch.size", len(batch.Samples)),
        ),
    )
    defer span.End()
    
    // Process batch without individual sample spans
    loss, err := nn.processBatch(ctx, batch)
    if err != nil {
        span.RecordError(err)
        return err
    }
    
    span.SetAttributes(attribute.Float64("batch.loss", loss))
    return nil
}
```

### Best Practices Summary

1. **Always propagate context** - Pass `context.Context` through all function calls
2. **Use meaningful span names** - Follow the format "component.operation"
3. **Add relevant attributes** - Include key parameters and results
4. **Record errors properly** - Use `span.RecordError()` and set error status
5. **Create span hierarchies** - Use child spans for sub-operations
6. **Be mindful of performance** - Use sampling for high-frequency operations
7. **Test your traces** - Verify trace creation and context propagation
8. **Use structured attributes** - Prefer semantic attributes over free-form text
9. **Correlate logs with traces** - Include trace ID in log messages
10. **Monitor trace overhead** - Use profiling to ensure observability doesn't hurt performance

---

## Additional Best Practices

### Context Usage

Always use context for cancellation and timeouts:

```go
// Good: Context-aware training
func (a *Agent) TrainWithContext(ctx context.Context, env Environment, episodes int) error {
    for i := 0; i < episodes; i++ {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        episode, err := a.RunEpisodeWithContext(ctx, env)
        if err != nil {
            return fmt.Errorf("episode %d failed: %w", i, err)
        }
        
        a.logProgress(i, episode)
    }
    return nil
}

func (a *Agent) RunEpisodeWithContext(ctx context.Context, env Environment) (*Episode, error) {
    state, err := env.Reset()
    if err != nil {
        return nil, err
    }
    
    episode := &Episode{}
    
    for {
        select {
        case <-ctx.Done():
            return episode, ctx.Err()
        default:
        }
        
        // ... episode logic
    }
}
```

### Logging

Use structured logging for agent systems:

```go
// Good: Structured logging
import "log/slog"

func (a *Agent) Train(env Environment, episodes int) error {
    logger := slog.With(
        "agent_id", a.ID,
        "environment", env.Name(),
        "total_episodes", episodes,
    )
    
    logger.Info("Starting training")
    
    for i := 0; i < episodes; i++ {
        episode, err := a.RunEpisode(env)
        if err != nil {
            logger.Error("Episode failed", 
                "episode", i,
                "error", err,
            )
            continue
        }
        
        logger.Info("Episode completed",
            "episode", i,
            "reward", episode.TotalReward,
            "steps", len(episode.Steps),
            "exploration_rate", a.ExplorationRate,
        )
        
        if i%100 == 0 {
            logger.Info("Training progress",
                "episodes_completed", i,
                "average_reward", a.GetAverageReward(),
                "success_rate", a.GetSuccessRate(),
            )
        }
    }
    
    logger.Info("Training completed")
    return nil
}
```

### Resource Management

Always clean up resources properly:

```go
// Good: Proper resource management
type Agent struct {
    brain    *brain.Network
    memory   *memory.Buffer
    sensors  []Sensor
    cancel   context.CancelFunc
    wg       sync.WaitGroup
}

func (a *Agent) Start(ctx context.Context) error {
    ctx, cancel := context.WithCancel(ctx)
    a.cancel = cancel
    
    // Start sensor goroutines
    for _, sensor := range a.sensors {
        a.wg.Add(1)
        go func(s Sensor) {
            defer a.wg.Done()
            if err := s.Run(ctx); err != nil {
                log.Printf("Sensor error: %v", err)
            }
        }(sensor)
    }
    
    return nil
}

func (a *Agent) Stop() error {
    if a.cancel != nil {
        a.cancel()
    }
    
    a.wg.Wait()
    
    // Clean up resources
    if err := a.brain.Close(); err != nil {
        return fmt.Errorf("failed to close brain: %w", err)
    }
    
    if err := a.memory.Close(); err != nil {
        return fmt.Errorf("failed to close memory: %w", err)
    }
    
    for _, sensor := range a.sensors {
        if err := sensor.Close(); err != nil {
            log.Printf("Failed to close sensor: %v", err)
        }
    }
    
    return nil
}
```

---

## Summary

This comprehensive guide provides the foundation for writing clear, maintainable, and efficient Go code for AI agents and intelligent systems. Key principles to remember:

1. **Clarity above all** - Code should be immediately understandable
2. **Handle errors explicitly** - Never ignore errors, always propagate them properly
3. **Use interfaces for testability** - Abstract external dependencies
4. **Manage goroutine lifetimes** - Always provide clear start/stop mechanisms
5. **Structure code logically** - Organize packages by responsibility
6. **Document thoroughly** - Especially complex algorithms and public APIs
7. **Test comprehensively** - Use table-driven tests and mocks
8. **Consider performance** - But not at the expense of clarity

Following these guidelines will result in agent systems that are not only functional but also maintainable, testable, and understandable by other developers.

---

*This guide is based on and adapts Google's Go Style Guide for AI agents development. For the complete original guide, see: https://google.github.io/styleguide/go/*
