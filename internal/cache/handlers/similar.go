package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raja-aiml/sematic-cache/internal/cache"
	"github.com/raja-aiml/sematic-cache/internal/storage"
)

// HandleSimilar finds similar cached responses
func HandleSimilar(backend storage.CacheBackend) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req cache.SimilarRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if req.TopK == 0 {
			req.TopK = 5
		}

		var results []cache.QueryResult
		if len(req.Embedding) > 0 {
			results = backend.GetTopKByEmbedding(req.Embedding, req.TopK)
		} else if req.Prompt != "" {
			var err error
			results, err = backend.GetTopKByText(c.Request.Context(), req.Prompt, req.TopK)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		} else {
			c.JSON(http.StatusBadRequest, gin.H{"error": "either prompt or embedding required"})
			return
		}

		c.JSON(http.StatusOK, cache.SimilarResponse{
			Prompt:  req.Prompt,
			Results: results,
		})
	}
}
