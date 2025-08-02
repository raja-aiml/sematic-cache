package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
	"github.com/raja-aiml/sematic-cache/pkg/agent"
)

// AgentHandler handles natural language requests to the memory agent
type AgentHandler struct {
	agent *agent.SemanticMemoryAgent
}

// NewAgentHandler creates a new agent handler
func NewAgentHandler(store *storage.VectorStore, embedFunc cache.EmbeddingFunc) *AgentHandler {
	return &AgentHandler{
		agent: agent.NewSemanticMemoryAgent(store, embedFunc),
	}
}

// HandleAgentRequest processes natural language requests
// POST /agent/process
func (h *AgentHandler) HandleAgentRequest(c *gin.Context) {
	var req agent.MemoryRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request format",
			"details": err.Error(),
		})
		return
	}

	// Process the natural language request
	response, err := h.agent.Process(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Agent processing failed",
			"details": err.Error(),
		})
		return
	}

	// Return appropriate status code based on success
	status := http.StatusOK
	if !response.Success {
		status = http.StatusUnprocessableEntity
	}

	c.JSON(status, response)
}

// HandleAgentChat provides a simple chat interface
// POST /agent/chat
func (h *AgentHandler) HandleAgentChat(c *gin.Context) {
	var input struct {
		Message string `json:"message" binding:"required"`
	}

	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Please provide a message",
		})
		return
	}

	// Create request from chat message
	req := agent.MemoryRequest{
		Query: input.Message,
	}

	// Process through agent
	response, err := h.agent.Process(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"reply": "I encountered an error processing your request.",
			"error": err.Error(),
		})
		return
	}

	// Format as chat response
	c.JSON(http.StatusOK, gin.H{
		"reply":  response.Message,
		"data":   response.Data,
		"intent": response.Intent,
	})
}

// AddAgentRoutes adds agent-related routes to the router
func AddAgentRoutes(router *gin.Engine, store *storage.VectorStore, embedFunc cache.EmbeddingFunc) {
	handler := NewAgentHandler(store, embedFunc)

	// Agent API routes
	agentGroup := router.Group("/agent")
	{
		agentGroup.POST("/process", handler.HandleAgentRequest)
		agentGroup.POST("/chat", handler.HandleAgentChat)
	}
}
