package handlers

import (
	"fmt"
	"math/rand"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/pkg/agent"
)

// A2AHandler provides HTTP endpoints for A2A protocol memory operations
type A2AHandler struct {
	adapter *agent.A2AMemoryAdapter
}

// NewA2AHandler creates a new A2A protocol handler
func NewA2AHandler(adapter *agent.A2AMemoryAdapter) *A2AHandler {
	return &A2AHandler{adapter: adapter}
}

// RegisterRoutes registers A2A protocol routes
func (h *A2AHandler) RegisterRoutes(router *gin.RouterGroup) {
	a2a := router.Group("/a2a")
	{
		// Memory operations
		a2a.POST("/memory/store", h.StoreMemory)
		a2a.GET("/memory/:agent_id/:message_id", h.RetrieveMemory)
		a2a.POST("/memory/search", h.SearchMemory)
		a2a.DELETE("/memory/:agent_id/forget", h.ForgetMemory)
		
		// Conversation management
		a2a.GET("/conversation/:agent_id/history", h.GetConversationHistory)
		a2a.POST("/context/relevant", h.GetRelevantContext)
		
		// Agent registration and management
		a2a.POST("/agent/register", h.RegisterAgent)
		a2a.GET("/agent/:agent_id/status", h.GetAgentStatus)
	}
}

// StoreMemory stores a new memory for an agent
func (h *A2AHandler) StoreMemory(c *gin.Context) {
	var msg agent.A2AMessage
	if err := c.ShouldBindJSON(&msg); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Set timestamp if not provided
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now()
	}
	
	// Generate ID if not provided
	if msg.ID == "" {
		msg.ID = generateMessageID()
	}
	
	if err := h.adapter.Store(c.Request.Context(), &msg); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"id": msg.ID,
		"status": "stored",
		"memory_type": msg.MemoryType,
	})
}

// RetrieveMemory retrieves a specific memory by ID
func (h *A2AHandler) RetrieveMemory(c *gin.Context) {
	agentID := c.Param("agent_id")
	messageID := c.Param("message_id")
	
	msg, err := h.adapter.Retrieve(c.Request.Context(), agentID, messageID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "memory not found"})
		return
	}
	
	c.JSON(http.StatusOK, msg)
}

// SearchMemory performs semantic search across memories
func (h *A2AHandler) SearchMemory(c *gin.Context) {
	var req struct {
		Query      string            `json:"query" binding:"required"`
		AgentID    string            `json:"agent_id"`
		MemoryType agent.MemoryType  `json:"memory_type"`
		TopK       int               `json:"top_k"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.TopK == 0 {
		req.TopK = 10
	}
	
	results, err := h.adapter.Search(c.Request.Context(), req.Query, req.AgentID, req.MemoryType, req.TopK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"query": req.Query,
		"count": len(results),
		"memories": results,
	})
}

// ForgetMemory removes old memories for an agent
func (h *A2AHandler) ForgetMemory(c *gin.Context) {
	agentID := c.Param("agent_id")
	olderThanStr := c.Query("older_than")
	
	olderThan := 24 * time.Hour // Default to 24 hours
	if olderThanStr != "" {
		if duration, err := time.ParseDuration(olderThanStr); err == nil {
			olderThan = duration
		}
	}
	
	if err := h.adapter.Forget(c.Request.Context(), agentID, olderThan); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"agent_id": agentID,
		"status": "forgotten",
		"older_than": olderThan.String(),
	})
}

// GetConversationHistory retrieves recent conversation history
func (h *A2AHandler) GetConversationHistory(c *gin.Context) {
	agentID := c.Param("agent_id")
	limitStr := c.DefaultQuery("limit", "20")
	limit, _ := strconv.Atoi(limitStr)
	
	if limit == 0 {
		limit = 20
	}
	
	history, err := h.adapter.GetConversationHistory(c.Request.Context(), agentID, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"agent_id": agentID,
		"count": len(history),
		"history": history,
	})
}

// GetRelevantContext retrieves contextually relevant memories
func (h *A2AHandler) GetRelevantContext(c *gin.Context) {
	var req struct {
		Query   string `json:"query" binding:"required"`
		AgentID string `json:"agent_id"`
		Limit   int    `json:"limit"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.Limit == 0 {
		req.Limit = 10
	}
	
	context, err := h.adapter.GetRelevantContext(c.Request.Context(), req.Query, req.AgentID, req.Limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"query": req.Query,
		"count": len(context),
		"context": context,
	})
}

// RegisterAgent registers a new agent in the system
func (h *A2AHandler) RegisterAgent(c *gin.Context) {
	var req struct {
		AgentID     string                 `json:"agent_id" binding:"required"`
		Name        string                 `json:"name"`
		Capabilities []string              `json:"capabilities"`
		Metadata    map[string]interface{} `json:"metadata"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Store agent registration as a special memory
	msg := &agent.A2AMessage{
		ID:         generateMessageID(),
		AgentID:    req.AgentID,
		Timestamp:  time.Now(),
		Type:       "agent_registration",
		Content:    req.Name,
		Context:    req.Metadata,
		MemoryType: agent.LongTermMemory,
		Importance: 1.0,
	}
	
	if err := h.adapter.Store(c.Request.Context(), msg); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"agent_id": req.AgentID,
		"status": "registered",
		"timestamp": msg.Timestamp,
	})
}

// GetAgentStatus retrieves the status of an agent
func (h *A2AHandler) GetAgentStatus(c *gin.Context) {
	agentID := c.Param("agent_id")
	
	// Get recent activity
	history, _ := h.adapter.GetConversationHistory(c.Request.Context(), agentID, 1)
	
	lastActive := time.Time{}
	if len(history) > 0 {
		lastActive = history[0].Timestamp
	}
	
	c.JSON(http.StatusOK, gin.H{
		"agent_id": agentID,
		"status": "active",
		"last_active": lastActive,
		"memory_count": len(history),
	})
}

// Helper function to generate unique message IDs
func generateMessageID() string {
	return fmt.Sprintf("msg_%d_%d", time.Now().UnixNano(), rand.Int63())
}