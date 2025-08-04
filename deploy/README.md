# Sentio RAG System - Production Deployment

This directory contains production-ready Kubernetes manifests for deploying the Sentio RAG system.

## Architecture Overview

The deployment consists of:

- **Sentio RAG API**: Main application with 3 replicas for high availability
- **Qdrant**: Vector database for document embeddings (StatefulSet with persistent storage)
- **Redis**: Cache layer for improved performance
- **Ingress**: NGINX-based ingress with SSL termination and security headers
- **Monitoring**: Prometheus metrics collection and alerting
- **Autoscaling**: Horizontal and Vertical Pod Autoscalers
- **Security**: RBAC, Network Policies, Pod Security Standards

## Prerequisites

1. **Kubernetes Cluster** (v1.24+)
2. **NGINX Ingress Controller**
3. **Cert-Manager** (for SSL certificates)
4. **Prometheus Operator** (for monitoring)
5. **Metrics Server** (for HPA)
6. **Container Image** built and pushed to your registry

## Quick Start

1. **Update Configuration**
   ```bash
   # Edit the following files with your specific values:
   # - secrets.yaml: Add base64-encoded API keys
   # - configmap.yaml: Update CORS origins and other settings
   # - ingress.yaml: Replace "your-domain.com" with your actual domain
   # - kustomization.yaml: Update image tags and versions
   ```

2. **Deploy with Kustomize**
   ```bash
   kubectl apply -k deploy/kubernetes/
   ```

3. **Verify Deployment**
   ```bash
   kubectl get pods -n sentio-rag
   kubectl get services -n sentio-rag
   kubectl logs -f deployment/sentio-rag -n sentio-rag
   ```

## Manual Deployment (without Kustomize)

If you prefer to deploy manually:

```bash
# Create namespace and basic resources
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f rbac.yaml

# Deploy data layer
kubectl apply -f redis.yaml
kubectl apply -f qdrant.yaml

# Wait for data layer to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n sentio-rag --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant -n sentio-rag --timeout=300s

# Deploy application
kubectl apply -f sentio-rag.yaml

# Wait for application to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=sentio-rag -n sentio-rag --timeout=300s

# Deploy networking and policies
kubectl apply -f ingress.yaml
kubectl apply -f networkpolicy.yaml
kubectl apply -f pdb.yaml

# Deploy autoscaling
kubectl apply -f hpa.yaml

# Deploy monitoring (requires Prometheus Operator)
kubectl apply -f monitoring.yaml
```

## Configuration

### Secrets Configuration

Before deployment, encode your secrets using base64:

```bash
echo -n "your-openai-api-key" | base64
echo -n "your-jina-api-key" | base64
echo -n "your-redis-password" | base64
# ... etc
```

Add the base64-encoded values to `secrets.yaml`.

### Domain Configuration

Update the following files with your domain:
- `ingress.yaml`: Replace `api.your-domain.com` and `metrics.your-domain.com`
- `configmap.yaml`: Update `CORS_ORIGINS` with your allowed origins

### Image Configuration

Update `kustomization.yaml` or `sentio-rag.yaml` with your actual image:
```yaml
image: your-registry.com/sentio-rag:3.0.0
```

## Monitoring and Observability

The deployment includes comprehensive monitoring:

### Metrics
- Application metrics exposed at `/metrics`
- Resource metrics via cAdvisor
- Custom business metrics for RAG operations

### Alerts
- Service availability
- Error rates
- Response latency
- Resource utilization
- Dependency health

### Health Checks
- Liveness probe: `/health/live`
- Readiness probe: `/health/ready`
- Startup probe: `/health`

## Security Features

### Network Security
- Network policies restricting inter-pod communication
- Ingress-level rate limiting and WAF rules
- TLS termination with secure headers

### Access Control
- RBAC with minimal permissions
- Service accounts with specific roles
- Pod Security Standards enforcement

### Secret Management
- Kubernetes secrets for sensitive data
- No secrets in environment variables or logs
- Encryption at rest (cluster-level)

## Scaling and Performance

### Horizontal Pod Autoscaler (HPA)
- CPU-based scaling (target: 70%)
- Memory-based scaling (target: 80%)
- Custom metrics scaling (requests/sec, response time)
- Min replicas: 3, Max replicas: 10

### Vertical Pod Autoscaler (VPA)
- Automatic resource recommendation and adjustment
- Memory: 256Mi - 4Gi
- CPU: 100m - 2000m

### Resource Limits
- Production-tuned resource requests and limits
- Memory limits prevent OOM kills
- CPU limits ensure fair scheduling

## High Availability

### Pod Distribution
- Anti-affinity rules spread pods across nodes
- Pod Disruption Budgets ensure minimum availability
- Multiple replicas handle node failures

### Data Persistence
- Qdrant uses persistent volumes for vector data
- Redis configured for cache with appropriate eviction policy
- StatefulSet ensures stable storage for Qdrant

### Load Balancing
- Service-level load balancing
- Ingress-level session affinity options
- Health checks ensure traffic to healthy pods

## Troubleshooting

### Common Issues

1. **Pods in Pending state**
   ```bash
   kubectl describe pod <pod-name> -n sentio-rag
   # Check for resource constraints, node selectors, or PVC issues
   ```

2. **Service not accessible**
   ```bash
   kubectl get svc -n sentio-rag
   kubectl describe ingress sentio-rag-ingress -n sentio-rag
   # Check ingress controller logs and DNS configuration
   ```

3. **Application errors**
   ```bash
   kubectl logs -f deployment/sentio-rag -n sentio-rag
   # Check for configuration issues, API key problems, or dependency failures
   ```

### Debug Commands

```bash
# Check all resources
kubectl get all -n sentio-rag

# Check events
kubectl get events -n sentio-rag --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n sentio-rag
kubectl top nodes

# Check ingress
kubectl get ingress -n sentio-rag -o wide

# Test internal connectivity
kubectl exec -it deployment/sentio-rag -n sentio-rag -- curl http://qdrant-service:6333/health
kubectl exec -it deployment/sentio-rag -n sentio-rag -- redis-cli -h redis-service ping
```

## Maintenance

### Updates
1. Update image tags in `kustomization.yaml`
2. Apply with rolling update: `kubectl apply -k deploy/kubernetes/`
3. Monitor deployment: `kubectl rollout status deployment/sentio-rag -n sentio-rag`

### Backup
- Qdrant data: Backup persistent volumes
- Configuration: Keep GitOps repository of manifests
- Secrets: Store securely in external secret management

### Monitoring
- Set up alerts for critical metrics
- Review resource usage regularly
- Monitor application-specific metrics

## Production Checklist

- [ ] Secrets properly configured and base64 encoded
- [ ] Domain names updated in ingress configuration
- [ ] Image tags point to tested and approved versions
- [ ] Resource limits appropriate for your workload
- [ ] Monitoring and alerting configured
- [ ] Backup strategy in place
- [ ] Security policies reviewed
- [ ] SSL certificates configured
- [ ] Rate limiting configured appropriately
- [ ] CORS origins set correctly
- [ ] Health checks tested
- [ ] Scaling policies configured
- [ ] Network policies tested
- [ ] Documentation updated for your environment