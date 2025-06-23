# Deployment Guide

This guide covers different deployment scenarios for the Codebase Indexing Solution.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- Git

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd codebase-indexing-solution
   chmod +x scripts/*.sh
   ./scripts/setup.sh
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Services**
   ```bash
   ./scripts/start.sh
   ```

4. **Index Sample Codebase**
   ```bash
   ./scripts/index-sample.sh
   ```

5. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Create Production Docker Compose**
   ```yaml
   # docker-compose.prod.yml
   version: '3.8'
   
   services:
     backend:
       build:
         context: ./backend
         dockerfile: Dockerfile
       ports:
         - "8000:8000"
       environment:
         - QDRANT_HOST=qdrant
         - NEO4J_URI=bolt://neo4j:7687
       depends_on:
         - qdrant
         - neo4j
       volumes:
         - ./data:/app/data
   
     frontend:
       build:
         context: ./frontend
         dockerfile: Dockerfile
       ports:
         - "3000:3000"
       environment:
         - REACT_APP_API_URL=http://localhost:8000
       depends_on:
         - backend
   
     qdrant:
       image: qdrant/qdrant:latest
       ports:
         - "6333:6333"
       volumes:
         - qdrant_data:/qdrant/storage
   
     neo4j:
       image: neo4j:5.15-community
       ports:
         - "7474:7474"
         - "7687:7687"
       environment:
         - NEO4J_AUTH=neo4j/your-secure-password
       volumes:
         - neo4j_data:/data
   
   volumes:
     qdrant_data:
     neo4j_data:
   ```

2. **Create Dockerfiles**

   **Backend Dockerfile:**
   ```dockerfile
   # backend/Dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Create non-root user
   RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
   USER appuser
   
   EXPOSE 8000
   
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

   **Frontend Dockerfile:**
   ```dockerfile
   # frontend/Dockerfile
   FROM node:18-alpine as builder
   
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci --only=production
   
   COPY . .
   RUN npm run build
   
   FROM nginx:alpine
   COPY --from=builder /app/build /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/nginx.conf
   
   EXPOSE 3000
   CMD ["nginx", "-g", "daemon off;"]
   ```

3. **Deploy**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Push Images to ECR**
   ```bash
   # Create ECR repositories
   aws ecr create-repository --repository-name codebase-indexing/backend
   aws ecr create-repository --repository-name codebase-indexing/frontend
   
   # Build and push images
   docker build -t codebase-indexing/backend ./backend
   docker build -t codebase-indexing/frontend ./frontend
   
   # Tag and push to ECR
   docker tag codebase-indexing/backend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/codebase-indexing/backend:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/codebase-indexing/backend:latest
   ```

2. **Create ECS Task Definition**
   ```json
   {
     "family": "codebase-indexing",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "backend",
         "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/codebase-indexing/backend:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "QDRANT_HOST",
             "value": "your-qdrant-host"
           }
         ]
       }
     ]
   }
   ```

#### Using Kubernetes

1. **Create Kubernetes Manifests**
   ```yaml
   # k8s/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: codebase-indexing-backend
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: codebase-indexing-backend
     template:
       metadata:
         labels:
           app: codebase-indexing-backend
       spec:
         containers:
         - name: backend
           image: codebase-indexing/backend:latest
           ports:
           - containerPort: 8000
           env:
           - name: QDRANT_HOST
             value: "qdrant-service"
           - name: NEO4J_URI
             value: "bolt://neo4j-service:7687"
   ```

2. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

### Google Cloud Platform

#### Using Cloud Run

1. **Deploy Backend**
   ```bash
   gcloud run deploy codebase-indexing-backend \
     --image gcr.io/PROJECT-ID/codebase-indexing/backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Deploy Frontend**
   ```bash
   gcloud run deploy codebase-indexing-frontend \
     --image gcr.io/PROJECT-ID/codebase-indexing/frontend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Production Considerations

### Security

1. **Environment Variables**
   - Use secure secret management (AWS Secrets Manager, Azure Key Vault, etc.)
   - Never commit API keys or passwords to version control
   - Use different credentials for different environments

2. **Network Security**
   - Use HTTPS/TLS for all communications
   - Implement proper firewall rules
   - Use VPC/private networks for database connections

3. **Authentication & Authorization**
   - Implement API authentication (JWT, OAuth2)
   - Add rate limiting
   - Use CORS properly

### Performance

1. **Database Optimization**
   - Configure Qdrant for production workloads
   - Optimize Neo4j memory settings
   - Use database connection pooling

2. **Caching**
   - Implement Redis for caching frequent queries
   - Use CDN for frontend assets
   - Cache embedding results

3. **Scaling**
   - Use load balancers for multiple backend instances
   - Implement horizontal scaling for databases
   - Use auto-scaling groups

### Monitoring

1. **Application Monitoring**
   ```python
   # Add to backend/src/monitoring.py
   from prometheus_client import Counter, Histogram, generate_latest
   
   REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
   REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
   ```

2. **Infrastructure Monitoring**
   - Use CloudWatch, Datadog, or Prometheus
   - Monitor CPU, memory, disk usage
   - Set up alerts for critical metrics

3. **Logging**
   ```python
   # Configure structured logging
   import structlog
   
   logger = structlog.get_logger()
   logger.info("Query processed", query=query, duration=duration, results=count)
   ```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database connectivity
   docker exec -it qdrant_container curl http://localhost:6333/collections
   docker exec -it neo4j_container cypher-shell -u neo4j -p password
   ```

2. **Memory Issues**
   - Increase Docker memory limits
   - Optimize batch sizes for indexing
   - Use streaming for large datasets

3. **Performance Issues**
   - Check database indexes
   - Monitor query performance
   - Optimize embedding model selection

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/mcp/status

# Frontend health
curl http://localhost:3000
```

### Logs

```bash
# View application logs
docker-compose logs -f backend
docker-compose logs -f frontend

# View database logs
docker-compose logs -f qdrant
docker-compose logs -f neo4j
```
