# MOSAIC API Specifications

## Overview

This document provides complete API specifications for the MOSAIC multi-tiered architecture. All APIs follow RESTful principles and use JSON for data exchange unless otherwise specified.

## Table of Contents

1. [Authentication & Security](#authentication--security)
2. [Core APIs](#core-apis)
3. [WebSocket APIs](#websocket-apis)
4. [GraphQL API](#graphql-api)
5. [Internal Service APIs](#internal-service-apis)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)

## Authentication & Security

### Session Management

All API requests require authentication via ephemeral session tokens that rotate every 30-60 minutes.

```http
X-Session-Token: <ephemeral-token>
```

### Token Generation

```http
POST /api/v1/auth/session
Content-Type: application/json

{
  "device_id": "unique-device-identifier",
  "timestamp": 1234567890,
  "signature": "ed25519-signature"
}
```

**Response:**
```json
{
  "session_token": "ephemeral-token-string",
  "expires_at": 1234567890,
  "rotation_interval": 1800
}
```

## Core APIs

### 1. Broadcast API

Submit obfuscated location and context events.

```http
POST /api/v1/broadcast
Content-Type: application/json
X-Session-Token: <token>

{
  "events": [
    {
      "location": {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "accuracy": 200
      },
      "context": "Coffee meeting at downtown",
      "timestamp": 1234567890,
      "noise_ratio": 0.6
    }
  ],
  "client_version": "1.0.0"
}
```

**Response:**
```json
{
  "status": "accepted",
  "event_ids": ["event-id-1", "event-id-2"],
  "server_timestamp": 1234567890
}
```

**Validation Rules:**
- `noise_ratio` must be between 0.3 and 0.8
- `context` maximum 500 characters
- `events` array maximum 10 items
- `timestamp` must be within ±10 minutes of server time

### 2. Proximity Query API

Query nearby contexts with privacy-preserving fuzzy matching.

```http
GET /api/v1/proximity?lat=37.7749&lon=-122.4194&radius=500&time_window=7200
X-Session-Token: <token>
```

**Query Parameters:**
| Parameter | Type | Required | Description | Default | Constraints |
|-----------|------|----------|-------------|---------|-------------|
| lat | float | Yes | Latitude | - | -90 to 90 |
| lon | float | Yes | Longitude | - | -180 to 180 |
| radius | int | No | Search radius in meters | 500 | 100-5000 |
| time_window | int | No | Time window in seconds | 7200 | 300-86400 |
| limit | int | No | Maximum results | 50 | 1-100 |
| categories | string | No | Comma-separated categories | all | - |

**Response:**
```json
{
  "contexts": [
    {
      "id": "context-id",
      "text": "Coffee meeting downtown",
      "approximate_distance": 250,
      "time_ago": 1800,
      "category": "social",
      "relevance_score": 0.85
    }
  ],
  "query_timestamp": 1234567890,
  "fuzzy_radius": 550
}
```

### 3. Association API

#### Create Association Token

```http
POST /api/v1/association/create
Content-Type: application/json
X-Session-Token: <token>

{
  "ttl": 86400,
  "max_uses": 1
}
```

**Response:**
```json
{
  "token": "32-character-hex-string",
  "expires_at": 1234567890,
  "association_url": "mosaic://associate/32-character-hex-string"
}
```

#### Initiate Association

```http
POST /api/v1/association/initiate
Content-Type: application/json
X-Session-Token: <token>

{
  "token": "32-character-hex-string",
  "public_key": "base64-encoded-x25519-public-key",
  "metadata": {
    "nickname": "Anonymous User",
    "avatar_hash": "sha256-hash"
  }
}
```

**Response:**
```json
{
  "status": "pending",
  "remote_public_key": "base64-encoded-x25519-public-key",
  "session_id": "association-session-id"
}
```

#### Accept Association

```http
POST /api/v1/association/accept
Content-Type: application/json
X-Session-Token: <token>

{
  "token": "32-character-hex-string",
  "public_key": "base64-encoded-x25519-public-key"
}
```

**Response:**
```json
{
  "status": "accepted",
  "initiator_public_key": "base64-encoded-x25519-public-key",
  "session_id": "association-session-id",
  "encrypted_channel": {
    "endpoint": "wss://relay.mosaic.privacy/channel/xyz",
    "auth_token": "channel-auth-token"
  }
}
```

#### Exchange Contact

```http
POST /api/v1/association/exchange
Content-Type: application/json
X-Session-Token: <token>

{
  "session_id": "association-session-id",
  "encrypted_contact": "base64-encoded-chacha20-encrypted-data",
  "nonce": "base64-encoded-12-byte-nonce"
}
```

**Response:**
```json
{
  "remote_contact": "base64-encoded-chacha20-encrypted-data",
  "remote_nonce": "base64-encoded-12-byte-nonce",
  "exchange_timestamp": 1234567890
}
```

### 4. Context Management API

#### Create Context

```http
POST /api/v1/context
Content-Type: application/json
X-Session-Token: <token>

{
  "text": "Looking for hiking partners this weekend",
  "enrichment_level": 0.5,
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "ttl": 86400,
  "categories": ["outdoor", "social"]
}
```

**Response:**
```json
{
  "context_id": "uuid-v4",
  "original_text": "Looking for hiking partners this weekend",
  "enriched_text": "Seeking outdoor enthusiasts for weekend trail adventure",
  "created_at": 1234567890,
  "expires_at": 1234654290,
  "privacy_score": 0.7
}
```

#### Update Context

```http
PATCH /api/v1/context/{context_id}
Content-Type: application/json
X-Session-Token: <token>

{
  "text": "Updated context text",
  "ttl": 43200
}
```

**Response:**
```json
{
  "status": "updated",
  "modified_at": 1234567890
}
```

#### Delete Context

```http
DELETE /api/v1/context/{context_id}
X-Session-Token: <token>
```

**Response:**
```json
{
  "status": "deleted",
  "deleted_at": 1234567890
}
```

### 5. Privacy Settings API

#### Get Current Settings

```http
GET /api/v1/privacy/settings
X-Session-Token: <token>
```

**Response:**
```json
{
  "privacy_mode": "balanced",
  "noise_ratio": 0.6,
  "session_rotation_minutes": 45,
  "default_ttl_hours": 24,
  "tor_enabled": false,
  "dummy_traffic_enabled": true,
  "location_precision_meters": 200
}
```

#### Update Settings

```http
PUT /api/v1/privacy/settings
Content-Type: application/json
X-Session-Token: <token>

{
  "privacy_mode": "maximum",
  "noise_ratio": 0.8,
  "tor_enabled": true
}
```

**Response:**
```json
{
  "status": "updated",
  "applied_settings": {
    "privacy_mode": "maximum",
    "noise_ratio": 0.8,
    "tor_enabled": true
  },
  "effective_at": 1234567890
}
```

## WebSocket APIs

### Real-time Updates

Connect to receive real-time proximity updates and association requests.

```javascript
const ws = new WebSocket('wss://api.mosaic.privacy/v1/ws');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'session-token'
}));

// Subscribe to proximity updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'proximity',
  params: {
    location: { lat: 37.7749, lon: -122.4194 },
    radius: 500
  }
}));

// Receive updates
ws.on('message', (data) => {
  const message = JSON.parse(data);
  switch (message.type) {
    case 'proximity_update':
      // Handle new nearby contexts
      break;
    case 'association_request':
      // Handle incoming association
      break;
    case 'privacy_alert':
      // Handle privacy warnings
      break;
  }
});
```

### Message Types

#### Proximity Update
```json
{
  "type": "proximity_update",
  "contexts": [...],
  "timestamp": 1234567890
}
```

#### Association Request
```json
{
  "type": "association_request",
  "token": "association-token",
  "metadata": {...},
  "expires_in": 3600
}
```

#### Privacy Alert
```json
{
  "type": "privacy_alert",
  "severity": "warning",
  "message": "Unusual query pattern detected",
  "recommended_action": "increase_noise_ratio"
}
```

## GraphQL API

### Schema

```graphql
type Query {
  # Proximity search
  nearbyContexts(
    location: LocationInput!
    radius: Float = 500
    timeWindow: Int = 7200
    limit: Int = 50
  ): [Context!]!

  # Association management
  associations(
    status: AssociationStatus
    limit: Int = 20
  ): [Association!]!

  # Privacy metrics
  privacyMetrics(
    window: TimeWindow!
  ): PrivacyMetrics!

  # System status
  systemStatus: SystemStatus!
}

type Mutation {
  # Broadcasting
  broadcast(
    events: [EventInput!]!
  ): BroadcastResult!

  # Association operations
  createAssociation(
    ttl: Int = 86400
  ): AssociationToken!

  initiateAssociation(
    token: String!
    publicKey: String!
  ): AssociationSession!

  # Context management
  createContext(
    input: ContextInput!
  ): Context!

  updateContext(
    id: ID!
    updates: ContextUpdate!
  ): Context!

  # Privacy settings
  updatePrivacySettings(
    settings: PrivacySettingsInput!
  ): PrivacySettings!
}

type Subscription {
  # Real-time proximity updates
  proximityUpdates(
    location: LocationInput!
    radius: Float!
  ): Context!

  # Association requests
  associationRequests: AssociationRequest!

  # Privacy alerts
  privacyAlerts: PrivacyAlert!
}

# Input Types
input LocationInput {
  latitude: Float!
  longitude: Float!
  accuracy: Float
}

input EventInput {
  location: LocationInput!
  context: String!
  noiseRatio: Float!
}

input ContextInput {
  text: String!
  enrichmentLevel: Float
  location: LocationInput
  ttl: Int
  categories: [String!]
}

input PrivacySettingsInput {
  privacyMode: PrivacyMode
  noiseRatio: Float
  torEnabled: Boolean
  dummyTrafficEnabled: Boolean
}

# Object Types
type Context {
  id: ID!
  text: String!
  location: Location
  distance: Float
  timestamp: Int!
  category: String
  score: Float
}

type Association {
  id: ID!
  token: String!
  status: AssociationStatus!
  createdAt: Int!
  expiresAt: Int!
  metadata: JSON
}

type PrivacyMetrics {
  noiseRatio: Float!
  sessionsRotated: Int!
  eventsObfuscated: Int!
  privacyScore: Float!
}

# Enums
enum PrivacyMode {
  EMERGENCY
  BALANCED
  MAXIMUM
}

enum AssociationStatus {
  PENDING
  ACCEPTED
  EXPIRED
  CANCELLED
}

enum TimeWindow {
  HOUR
  DAY
  WEEK
  MONTH
}
```

### Example Queries

```graphql
# Search nearby contexts
query SearchNearby {
  nearbyContexts(
    location: { latitude: 37.7749, longitude: -122.4194 }
    radius: 1000
    timeWindow: 3600
  ) {
    id
    text
    distance
    timestamp
    category
  }
}

# Create broadcast
mutation Broadcast {
  broadcast(
    events: [{
      location: {
        latitude: 37.7749,
        longitude: -122.4194,
        accuracy: 200
      }
      context: "Coffee meetup downtown"
      noiseRatio: 0.6
    }]
  ) {
    status
    eventIds
  }
}

# Subscribe to updates
subscription ProximityUpdates {
  proximityUpdates(
    location: { latitude: 37.7749, longitude: -122.4194 }
    radius: 500
  ) {
    id
    text
    distance
  }
}
```

## Internal Service APIs

### Inter-Service Communication

Internal services use gRPC for efficient communication.

#### Protocol Buffers Definition

```protobuf
syntax = "proto3";
package mosaic.internal;

service CoordinationService {
  rpc StoreEvent(StoreEventRequest) returns (StoreEventResponse);
  rpc QueryProximity(QueryProximityRequest) returns (QueryProximityResponse);
  rpc FederateQuery(FederateQueryRequest) returns (FederateQueryResponse);
}

message StoreEventRequest {
  string event_id = 1;
  bytes encrypted_data = 2;
  repeated SecretShare shares = 3;
  int64 ttl_seconds = 4;
}

message SecretShare {
  int32 index = 1;
  bytes data = 2;
  string provider_id = 3;
}

message StoreEventResponse {
  bool success = 1;
  string error_message = 2;
  repeated string shard_ids = 3;
}

message QueryProximityRequest {
  Location location = 1;
  double radius_meters = 2;
  int64 time_window_seconds = 3;
  int32 max_results = 4;
}

message Location {
  double latitude = 1;
  double longitude = 2;
}

message QueryProximityResponse {
  repeated Context contexts = 1;
  int64 query_timestamp = 2;
}

message Context {
  string id = 1;
  string text = 2;
  double approximate_distance = 3;
  int64 timestamp = 4;
}
```

### Service Discovery

Services register with Consul for discovery.

```http
PUT /v1/agent/service/register
Content-Type: application/json

{
  "ID": "coordination-service-1",
  "Name": "coordination",
  "Tags": ["primary", "v1"],
  "Address": "10.1.1.100",
  "Port": 8500,
  "Check": {
    "HTTP": "http://10.1.1.100:8500/health",
    "Interval": "10s"
  }
}
```

## Error Handling

### Standard Error Response

All errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_LOCATION",
    "message": "Location coordinates are outside valid range",
    "details": {
      "field": "latitude",
      "value": 91.5,
      "constraint": "must be between -90 and 90"
    }
  },
  "request_id": "req-uuid-v4",
  "timestamp": 1234567890
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_SESSION | 401 | Session token invalid or expired |
| RATE_LIMITED | 429 | Too many requests |
| INVALID_LOCATION | 400 | Invalid location coordinates |
| NOISE_RATIO_LOW | 400 | Noise ratio below minimum threshold |
| CONTEXT_TOO_LONG | 400 | Context exceeds maximum length |
| ASSOCIATION_EXPIRED | 410 | Association token has expired |
| PRIVACY_VIOLATION | 403 | Request violates privacy constraints |
| SERVICE_UNAVAILABLE | 503 | Temporary service outage |
| FEDERATION_ERROR | 502 | Federated provider error |

### Retry Strategy

```http
Retry-After: 60
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1234567890
```

## Rate Limiting

### Default Limits

| Endpoint | Limit | Window | Burst |
|----------|-------|--------|-------|
| /broadcast | 10 | 1 hour | 3 |
| /proximity | 100 | 1 hour | 20 |
| /association/* | 20 | 1 hour | 5 |
| /context/* | 50 | 1 hour | 10 |
| WebSocket | 1000 msgs | 1 hour | 100 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1234567890
X-RateLimit-Reset-After: 1800
X-RateLimit-Bucket: proximity-query
```

### Adaptive Rate Limiting

Rate limits adjust based on:
- Privacy mode (higher privacy = higher limits)
- Proof of work completion
- Network congestion
- Abuse detection score

```json
{
  "current_limits": {
    "broadcast": 15,
    "proximity": 150
  },
  "modifiers": {
    "privacy_mode_bonus": 1.5,
    "proof_of_work": true,
    "reputation_score": 0.95
  },
  "next_adjustment": 1234567890
}
```

## API Versioning

### Version Strategy

APIs are versioned using URL path versioning:
- Current: `/api/v1/*`
- Beta: `/api/v2-beta/*`
- Deprecated: `/api/v0/*` (sunset date: 2025-12-31)

### Version Negotiation

```http
Accept: application/vnd.mosaic.v1+json
```

### Deprecation Policy

1. Announcement: 6 months before deprecation
2. Warning headers: 3 months before deprecation
3. Sunset header: 1 month before deprecation
4. Removal: After sunset date

```http
Warning: 299 - "API version v0 will be deprecated on 2025-12-31"
Sunset: Sat, 31 Dec 2025 23:59:59 GMT
```

## SDK Support

Official SDKs are available for:
- JavaScript/TypeScript (Node.js, Browser, React Native)
- Swift (iOS)
- Kotlin (Android)
- Go (Server)
- Python (Server)
- Rust (Server)

### SDK Example (TypeScript)

```typescript
import { MosaicClient } from '@mosaic/sdk';

const client = new MosaicClient({
  endpoint: 'https://api.mosaic.privacy',
  sessionToken: 'your-session-token',
  privacyMode: 'balanced'
});

// Broadcast location
await client.broadcast({
  location: { latitude: 37.7749, longitude: -122.4194 },
  context: 'Coffee meeting',
  noiseRatio: 0.6
});

// Query nearby
const contexts = await client.queryNearby({
  location: { latitude: 37.7749, longitude: -122.4194 },
  radius: 500,
  timeWindow: 7200
});

// Real-time updates
client.on('proximity:update', (contexts) => {
  console.log('New contexts:', contexts);
});

client.on('association:request', (request) => {
  console.log('Association requested:', request);
});
```

## Testing

### Test Environment

Test API endpoint: `https://api-test.mosaic.privacy`

Test credentials:
```json
{
  "device_id": "test-device-001",
  "api_key": "test-key-do-not-use-in-production"
}
```

### Postman Collection

Import the [MOSAIC API Postman Collection](https://api.mosaic.privacy/postman-collection.json) for interactive testing.

### cURL Examples

```bash
# Get session token
curl -X POST https://api.mosaic.privacy/v1/auth/session \
  -H "Content-Type: application/json" \
  -d '{"device_id":"test-device","timestamp":1234567890,"signature":"..."}'

# Broadcast event
curl -X POST https://api.mosaic.privacy/v1/broadcast \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: your-token" \
  -d '{"events":[{"location":{"latitude":37.7749,"longitude":-122.4194},"context":"Test","noise_ratio":0.6}]}'

# Query proximity
curl -G https://api.mosaic.privacy/v1/proximity \
  -H "X-Session-Token: your-token" \
  -d lat=37.7749 \
  -d lon=-122.4194 \
  -d radius=500
```

---

This API specification provides comprehensive documentation for all MOSAIC system APIs, enabling developers to integrate with the platform while maintaining privacy and security requirements.