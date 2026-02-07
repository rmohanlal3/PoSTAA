#!/bin/bash

# Deployment script for GCP
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-development}
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
CLUSTER_NAME="motivational-ai-cluster"

echo "🚀 Deploying Motivational App to GCP"
echo "Environment: $ENVIRONMENT"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Authenticate with GCP
echo "📝 Authenticating with GCP..."
gcloud auth login
gcloud config set project $PROJECT_ID

# Build and push backend container
echo "🐳 Building backend Docker image..."
cd ../backend
docker build -t gcr.io/$PROJECT_ID/motivational-backend:latest .
docker push gcr.io/$PROJECT_ID/motivational-backend:latest

# Get GKE credentials
echo "☸️  Getting GKE credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

# Create namespace if it doesn't exist
echo "📦 Creating namespace..."
kubectl create namespace motivational-app --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
echo "🔐 Creating secrets..."
kubectl create secret generic app-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=secret-key="your-secret-key" \
  --namespace=motivational-app \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy backend
echo "🚢 Deploying backend service..."
kubectl apply -f k8s/backend-deployment.yaml

# Deploy Riva (if needed)
if [ "$ENVIRONMENT" == "production" ]; then
  echo "🎤 Deploying Riva service..."
  kubectl apply -f k8s/riva-deployment.yaml
fi

# Deploy NeMo (if needed)
if [ "$ENVIRONMENT" == "production" ]; then
  echo "🧠 Deploying NeMo service..."
  kubectl apply -f k8s/nemo-deployment.yaml
fi

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/backend-api -n motivational-app

# Get external IP
echo "🌐 Getting external IP..."
EXTERNAL_IP=$(kubectl get service backend-api-service -n motivational-app -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "✅ Deployment complete!"
echo "Backend API: http://$EXTERNAL_IP"
echo ""
echo "Useful commands:"
echo "  View pods: kubectl get pods -n motivational-app"
echo "  View logs: kubectl logs -f deployment/backend-api -n motivational-app"
echo "  Scale deployment: kubectl scale deployment backend-api --replicas=5 -n motivational-app"
