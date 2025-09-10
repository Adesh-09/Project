# System Monitoring Pipeline - Deployment Guide

## Overview

This document provides comprehensive instructions for deploying and operating the System Monitoring Pipeline, which includes Spark-based anomaly detection, Prometheus monitoring, Grafana visualization, and proactive mitigation capabilities.

## Architecture Components

1. **Spark Cluster**: Processes system metrics and performs anomaly detection
2. **Prometheus**: Collects and stores time-series metrics
3. **Grafana**: Provides visualization dashboards
4. **Push Gateway**: Receives metrics from Spark applications
5. **Alertmanager**: Handles alert routing and notifications
6. **Redis**: Caches results and provides fast data access
7. **Monitoring API**: Flask-based API for dashboard integration
8. **Proactive Mitigation Engine**: Automatically responds to detected anomalies

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04+ or CentOS 8+
- **Memory**: Minimum 16GB RAM (32GB recommended for production)
- **CPU**: Minimum 8 cores (16 cores recommended for production)
- **Storage**: Minimum 100GB SSD storage
- **Network**: Stable internet connection for downloading dependencies

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- Java 11+
- Git

## Deployment Options

### Option 1: Docker Compose Deployment (Recommended for Development/Testing)

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd system_monitoring_pipeline
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your specific configurations
   ```

3. **Start the monitoring stack:**
   ```bash
   cd docker
   docker-compose up -d
   ```

4. **Verify deployment:**
   ```bash
   # Check all services are running
   docker-compose ps
   
   # Check logs
   docker-compose logs -f
   ```

5. **Access services:**
   - Grafana: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090
   - Spark Master UI: http://localhost:8080
   - Monitoring API: http://localhost:8000
   - Push Gateway: http://localhost:9091

### Option 2: AWS EC2 Deployment (Production)

#### Step 1: Launch EC2 Instances

1. **Launch EC2 instances:**
   - Instance type: m5.2xlarge or larger
   - AMI: Ubuntu 20.04 LTS
   - Security groups: Allow ports 22, 80, 443, 3000, 8080, 9090, 9091

2. **Configure security groups:**
   ```bash
   # Allow monitoring ports
   aws ec2 authorize-security-group-ingress \
     --group-id sg-xxxxxxxxx \
     --protocol tcp \
     --port 3000 \
     --cidr 0.0.0.0/0
   
   # Repeat for other ports: 8080, 9090, 9091, 8000
   ```

#### Step 2: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Java
sudo apt install openjdk-11-jdk -y

# Install Python dependencies
sudo apt install python3-pip -y
pip3 install -r requirements.txt
```

#### Step 3: Configure and Deploy

1. **Clone and configure:**
   ```bash
   git clone <repository_url>
   cd system_monitoring_pipeline
   
   # Configure for production
   cp config/production.env .env
   # Edit .env with production settings
   ```

2. **Deploy services:**
   ```bash
   cd docker
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. **Configure reverse proxy (optional):**
   ```bash
   # Install Nginx
   sudo apt install nginx -y
   
   # Configure Nginx for Grafana
   sudo cp config/nginx/grafana.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/grafana.conf /etc/nginx/sites-enabled/
   sudo systemctl reload nginx
   ```

## Configuration

### Prometheus Configuration

Edit `config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'system-metrics'
    static_configs:
      - targets: 
        - 'your-server-1:9100'
        - 'your-server-2:9100'
        # Add your actual server IPs
```

### Grafana Configuration

1. **Import dashboards:**
   ```bash
   # Copy dashboard JSON files
   cp grafana/dashboards/*.json /var/lib/grafana/dashboards/
   ```

2. **Configure data sources:**
   - Navigate to Configuration > Data Sources
   - Add Prometheus: http://prometheus:9090

### Alert Configuration

Edit `config/alert_rules.yml` to customize alert thresholds:

```yaml
groups:
  - name: custom_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage > 80  # Adjust threshold as needed
        for: 5m
        labels:
          severity: warning
```

## Operation and Maintenance

### Daily Operations

1. **Check system health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Monitor logs:**
   ```bash
   docker-compose logs -f monitoring-app
   docker-compose logs -f prometheus
   docker-compose logs -f grafana
   ```

3. **Check anomaly detection:**
   ```bash
   curl -X POST http://localhost:8000/api/anomalies/detect \
     -H "Content-Type: application/json" \
     -d '{"data_source": "sample"}'
   ```

### Weekly Maintenance

1. **Update system metrics:**
   ```bash
   # Restart Prometheus to reload configuration
   docker-compose restart prometheus
   ```

2. **Clean up old data:**
   ```bash
   # Clean Redis cache
   docker-compose exec redis redis-cli FLUSHDB
   
   # Clean old Prometheus data (if needed)
   docker-compose exec prometheus rm -rf /prometheus/old_data
   ```

3. **Backup configurations:**
   ```bash
   # Backup Grafana dashboards
   docker-compose exec grafana grafana-cli admin export-dashboard
   
   # Backup Prometheus data
   tar -czf prometheus_backup_$(date +%Y%m%d).tar.gz prometheus_data/
   ```

### Monthly Maintenance

1. **Update Docker images:**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

2. **Review and optimize alert rules:**
   - Analyze false positive rates
   - Adjust thresholds based on historical data
   - Update notification channels

3. **Performance tuning:**
   - Review Spark job performance
   - Optimize Prometheus retention policies
   - Clean up unused Grafana dashboards

## Monitoring and Alerting

### Key Metrics to Monitor

1. **System Performance:**
   - CPU usage across all servers
   - Memory utilization
   - Disk I/O and space usage
   - Network throughput

2. **Application Metrics:**
   - Anomaly detection accuracy
   - Mitigation success rate
   - API response times
   - Error rates

3. **Infrastructure Health:**
   - Spark job completion rates
   - Prometheus scrape success
   - Grafana dashboard load times
   - Redis connection status

### Alert Escalation

1. **Level 1 - Warning (Auto-mitigation):**
   - CPU usage > 70%
   - Memory usage > 70%
   - Response time > 500ms

2. **Level 2 - Critical (Immediate attention):**
   - CPU usage > 90%
   - Memory usage > 90%
   - Service unavailable
   - Multiple anomalies detected

3. **Level 3 - Emergency (Page operations team):**
   - System completely down
   - Data corruption detected
   - Security breach indicators

## Troubleshooting

### Common Issues

1. **Spark jobs failing:**
   ```bash
   # Check Spark logs
   docker-compose logs spark-master
   docker-compose logs spark-worker
   
   # Increase memory allocation
   # Edit docker-compose.yml and increase SPARK_WORKER_MEMORY
   ```

2. **Prometheus not scraping metrics:**
   ```bash
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   
   # Verify network connectivity
   docker-compose exec prometheus ping node-exporter
   ```

3. **Grafana dashboards not loading:**
   ```bash
   # Check Grafana logs
   docker-compose logs grafana
   
   # Verify data source configuration
   curl http://admin:admin123@localhost:3000/api/datasources
   ```

4. **High memory usage:**
   ```bash
   # Monitor container resource usage
   docker stats
   
   # Adjust memory limits in docker-compose.yml
   ```

### Performance Optimization

1. **Spark Optimization:**
   ```bash
   # Tune Spark configuration
   export SPARK_CONF_DIR=/opt/spark/conf
   # Edit spark-defaults.conf:
   # spark.sql.adaptive.enabled=true
   # spark.sql.adaptive.coalescePartitions.enabled=true
   ```

2. **Prometheus Optimization:**
   ```yaml
   # Adjust retention and storage
   command:
     - '--storage.tsdb.retention.time=30d'
     - '--storage.tsdb.retention.size=50GB'
   ```

3. **Redis Optimization:**
   ```bash
   # Configure Redis for better performance
   docker-compose exec redis redis-cli CONFIG SET maxmemory 2gb
   docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

## Security Considerations

1. **Network Security:**
   - Use VPC with private subnets
   - Configure security groups to allow only necessary ports
   - Enable SSL/TLS for all web interfaces

2. **Authentication:**
   - Change default passwords
   - Enable LDAP/OAuth integration for Grafana
   - Use API keys for programmatic access

3. **Data Protection:**
   - Encrypt data at rest
   - Enable audit logging
   - Regular security updates

## Scaling

### Horizontal Scaling

1. **Add more Spark workers:**
   ```bash
   docker-compose scale spark-worker=3
   ```

2. **Deploy multiple Prometheus instances:**
   - Use Prometheus federation
   - Implement sharding for large deployments

3. **Load balance Grafana:**
   - Deploy multiple Grafana instances
   - Use external database for session storage

### Vertical Scaling

1. **Increase resource limits:**
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 8G
         cpus: '4'
   ```

2. **Optimize JVM settings:**
   ```bash
   export SPARK_DRIVER_MEMORY=4g
   export SPARK_EXECUTOR_MEMORY=4g
   ```

## Backup and Recovery

### Backup Strategy

1. **Daily backups:**
   - Prometheus data
   - Grafana dashboards and configuration
   - Application logs

2. **Weekly backups:**
   - Complete system configuration
   - Docker images and containers
   - Historical anomaly data

3. **Monthly backups:**
   - Full system snapshot
   - Archive old data

### Recovery Procedures

1. **Service recovery:**
   ```bash
   # Restart failed services
   docker-compose restart <service_name>
   
   # Full stack restart
   docker-compose down && docker-compose up -d
   ```

2. **Data recovery:**
   ```bash
   # Restore Prometheus data
   tar -xzf prometheus_backup.tar.gz -C prometheus_data/
   
   # Restore Grafana dashboards
   cp backup/dashboards/*.json /var/lib/grafana/dashboards/
   ```

## Support and Maintenance Contacts

- **System Administrator**: admin@company.com
- **DevOps Team**: devops@company.com
- **On-call Engineer**: +1-555-0123
- **Emergency Escalation**: emergency@company.com

For additional support, refer to the project documentation or contact the development team.

