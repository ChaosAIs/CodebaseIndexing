# Enhanced Query Processing System

## Overview

The Enhanced Query Processing System implements a sophisticated **Embedding RAG + Graph RAG** pipeline that provides richer, more intelligent responses to codebase queries. This system goes beyond simple keyword matching to understand the semantic meaning, relationships, and architectural context of code.

## Architecture

### Core Components

1. **Advanced NER (Named Entity Recognition)**
   - spaCy-based entity extraction for general entities
   - Pattern-based extraction for code-specific entities
   - Knowledge base lookup for technology and company names
   - Confidence scoring and entity disambiguation

2. **Enhanced Query Processor**
   - Sophisticated query processing pipeline
   - Embedding-based semantic search
   - Graph-based relationship exploration
   - Advanced scoring and ranking algorithms

3. **Integration Layer**
   - Seamless integration with existing MCP server
   - Performance monitoring and statistics
   - Fallback mechanisms for reliability

## Query Processing Flow

### Step 1: Query Processing with NER
```
User Query → Advanced NER → Extracted Entities
```

**Example:**
- Query: "How does the FastAPI authentication work with JWT tokens?"
- Entities: FastAPI (framework), authentication (concept), JWT (technology)

### Step 2: Embedding Query
```
Query → Embedding Generation → Vector Search → Top-K Documents
```

**Features:**
- Multi-model embedding support (OpenAI, HuggingFace, Ollama)
- Cosine similarity search in Qdrant vector database
- Project-specific filtering

### Step 3: Graph Query
```
Entities → Graph Node Mapping → Relationship Traversal → Graph Context
```

**Cypher Query Example:**
```cypher
MATCH (c:Chunk)
WHERE c.node_type = 'function' AND c.name CONTAINS 'authenticate'
OPTIONAL MATCH (c)-[r1]->(neighbor1:Chunk)
OPTIONAL MATCH (c)<-[r2]-(neighbor2:Chunk)
RETURN c, neighbor1, neighbor2, r1, r2
```

### Step 4: Combine Results
```
Embedding Results + Graph Results → Weighted Scoring → Ranked Results
```

**Scoring Formula:**
```
final_score = w1 * embedding_score + w2 * graph_score + w3 * centrality_score + w4 * path_score
```

Default weights:
- `embedding_score`: 0.4
- `graph_score`: 0.3
- `centrality_score`: 0.15
- `path_score`: 0.15

### Step 5: Context Fusion
```
Ranked Results → Insight Generation → Combined Context → Rich Response
```

## Entity Types

### Code Entities
- **Function/Method**: `authenticate()`, `process_request()`
- **Class/Interface**: `UserService`, `AuthenticationManager`
- **File/Module**: `auth.py`, `user_service.js`
- **Variable/Constant**: `API_KEY`, `user_token`

### Technical Entities
- **Technology**: Python, JavaScript, React, FastAPI
- **Framework**: Django, Express, Spring Boot
- **Database**: PostgreSQL, MongoDB, Redis
- **Protocol/API**: REST, GraphQL, HTTP, JWT

### Business Entities
- **Person**: Tim Cook, Linus Torvalds
- **Organization**: Apple, Google, Microsoft
- **Concept**: authentication, scalability, performance

## Advanced Features

### 1. Centrality Analysis
Calculates node importance in the code graph:
```python
centrality_score = degree_count / max_degree
```

### 2. Path Analysis
Analyzes traversal paths between related code components:
```python
path_score = sum(1.0 / path_length for path in paths)
```

### 3. Architectural Layer Detection
Automatically identifies architectural layers:
- **API Layer**: Controllers, endpoints, routes
- **Business Layer**: Services, business logic
- **Data Layer**: Models, repositories, DAOs
- **Utility Layer**: Helpers, utilities, common functions

### 4. Context-Aware Scoring
Adjusts scores based on:
- Entity relevance to query
- Code relationship strength
- Architectural importance
- File-level significance

## Integration Examples

### Basic Usage
```python
from backend.src.query.enhanced_integration import EnhancedQueryIntegration

# Initialize
integration = EnhancedQueryIntegration(
    vector_store=vector_store,
    graph_store=graph_store,
    embedding_generator=embedding_generator
)

# Process query
results, metadata = await integration.process_query_enhanced(
    query="How does authentication work?",
    project_ids=["my-project"],
    limit=10
)
```

### With MCP Server
```python
from backend.src.query.integration_example import EnhancedMCPServer

# Enhanced server with sophisticated processing
server = EnhancedMCPServer(...)

# Enhanced search
response = await server.search_enhanced(search_request)

# Enhanced flow with multi-agent analysis
flow_response = await server.flow_enhanced(flow_request)
```

## Performance Optimizations

### 1. Smart Agent Selection
- Relevance-based agent filtering
- Dynamic threshold adjustment
- Early termination for low-relevance queries

### 2. Parallel Processing
- Concurrent entity extraction and graph queries
- Asynchronous embedding generation
- Parallel agent execution

### 3. Caching
- Query result caching
- Entity mapping cache
- Graph traversal cache

### 4. Fallback Mechanisms
- Graceful degradation to basic search
- Error handling with fallback responses
- Performance monitoring and alerts

## Configuration

### Scoring Weights
```python
scoring_weights = {
    "embedding_score": 0.4,    # Semantic similarity weight
    "graph_score": 0.3,        # Graph relationship weight
    "centrality_score": 0.15,  # Node importance weight
    "path_score": 0.15         # Path analysis weight
}
```

### NER Configuration
```python
# Enable/disable NER components
ner_config = {
    "use_spacy": True,           # Use spaCy for general NER
    "use_patterns": True,        # Use regex patterns for code entities
    "use_knowledge_base": True,  # Use knowledge base lookup
    "confidence_threshold": 0.5  # Minimum confidence for entities
}
```

## Monitoring and Analytics

### Performance Metrics
- Total queries processed
- Enhanced vs. basic processing ratio
- Average response time
- Average confidence score
- Entity extraction accuracy

### Query Analysis
- Complexity assessment
- Entity distribution
- Processing recommendations
- Success/failure rates

## Benefits

### 1. Richer Context Understanding
- Semantic understanding beyond keywords
- Relationship-aware search results
- Architectural context awareness

### 2. Better Result Quality
- Multi-perspective analysis
- Confidence-based ranking
- Context-relevant results

### 3. Enhanced User Experience
- More intelligent responses
- Detailed insights and explanations
- Adaptive processing based on query complexity

### 4. Scalable Architecture
- Modular design for easy extension
- Performance optimizations
- Robust error handling

## Future Enhancements

### 1. Machine Learning Integration
- Learned ranking models (XGBoost, neural networks)
- Query intent classification improvements
- Personalized result ranking

### 2. Advanced Graph Analysis
- Community detection algorithms
- Graph neural networks (GNN)
- Node2vec embeddings

### 3. Multi-Modal Processing
- Code visualization integration
- Documentation and comment analysis
- Version control history analysis

### 4. Real-Time Learning
- User feedback integration
- Adaptive scoring weights
- Continuous model improvement

## Conclusion

The Enhanced Query Processing System represents a significant advancement in codebase understanding and search capabilities. By combining sophisticated NER, embedding-based search, graph analysis, and intelligent scoring, it provides users with richer, more contextual, and more accurate responses to their codebase queries.

The system is designed to be both powerful and practical, with robust fallback mechanisms, performance optimizations, and seamless integration with existing infrastructure.
