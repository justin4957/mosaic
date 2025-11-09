# MOSAIC Threat Models
## Comprehensive Adversarial Analysis and Mitigation Strategies

**Version**: 1.0.0
**Date**: 2025-11-09
**Status**: Security Analysis Document
**Classification**: Public

---

## Table of Contents

1. [Adversary Classification](#adversary-classification)
2. [Threat Model 1: Passive Network Observer](#threat-model-1-passive-network-observer)
3. [Threat Model 2: Compromised Server](#threat-model-2-compromised-server)
4. [Threat Model 3: Statistical Correlation Attack](#threat-model-3-statistical-correlation-attack)
5. [Threat Model 4: Compelled Client Modification](#threat-model-4-compelled-client-modification)
6. [Threat Model 5: Social Graph Reconstruction](#threat-model-5-social-graph-reconstruction)
7. [Threat Model 6: Adversarial Machine Learning](#threat-model-6-adversarial-machine-learning)
8. [Threat Model 7: Collusion Attacks](#threat-model-7-collusion-attacks)
9. [Threat Model 8: Side-Channel Attacks](#threat-model-8-side-channel-attacks)
10. [Threat Model 9: Legal Compulsion](#threat-model-9-legal-compulsion)
11. [Threat Model 10: Sybil Attacks](#threat-model-10-sybil-attacks)
12. [Attack Trees](#attack-trees)
13. [Risk Matrix](#risk-matrix)

---

## Adversary Classification

### Capability Levels

**Level 1: Individual Attacker**
- Resources: Personal compute (<$1k/month)
- Access: Public internet, open-source tools
- Examples: Curious hacker, stalker

**Level 2: Organized Group**
- Resources: Moderate budget ($10k-$100k)
- Access: Commercial surveillance tools, botnets
- Examples: Private investigator, corporate competitor

**Level 3: Nation-State Actor**
- Resources: Substantial budget ($1M+)
- Access: SIGINT capabilities, legal compulsion, zero-days
- Examples: NSA, GCHQ, FSB, MSS

### Adversary Goals

1. **Deanonymization**: Link events to real-world identity
2. **Location Tracking**: Determine user's physical location over time
3. **Social Graph Mapping**: Identify relationships between users
4. **Content Surveillance**: Learn what users discuss/coordinate
5. **Denial of Service**: Prevent legitimate use of system

---

## Threat Model 1: Passive Network Observer

### Adversary Profile

**Capability**: Level 2-3
- **Access**: Can monitor all network traffic (ISP, government, wifi operator)
- **Cannot**: Decrypt TLS, modify packets, compromise endpoints
- **Goal**: Identify who is using MOSAIC and correlate activity patterns

### Attack Vectors

#### 1.1 Traffic Analysis

**Attack**: Identify MOSAIC usage by traffic patterns

```
Observable patterns:
- TLS connections to known MOSAIC servers
- Packet sizes (events ~2KB, queries ~500 bytes)
- Timing patterns (broadcast frequency)
- Connection metadata (IP addresses, timestamps)
```

**Impact**: HIGH
- Adversary learns user participates in MOSAIC
- Can correlate MOSAIC usage with other activity

**Mitigation**:
```python
class TrafficObfuscation:
    def obfuscate_traffic_pattern(self, real_request):
        """
        Make MOSAIC traffic indistinguishable from normal web traffic
        """
        # 1. Pad to constant size (mimic common HTTPS requests)
        padded_request = self.pad_to_size(real_request, target_size=4096)

        # 2. Add cover traffic (random timing)
        if random.random() < 0.3:  # 30% chance
            self.send_dummy_request()

        # 3. Batch multiple operations (hide individual actions)
        self.delay_and_batch(padded_request, max_delay_sec=30)

        return padded_request

    def tor_integration(self):
        """
        Optional: Route through Tor for IP anonymity
        """
        # Use Tor SOCKS5 proxy
        proxy = {
            'http': 'socks5h://localhost:9050',
            'https': 'socks5h://localhost:9050'
        }
        return requests.post(url, proxies=proxy)
```

**Residual Risk**: LOW
- With Tor + padding + cover traffic, MOSAIC traffic indistinguishable
- Adversary cannot determine MOSAIC usage with confidence

---

#### 1.2 Timing Correlation

**Attack**: Correlate broadcast times with real-world events

```
Scenario:
1. Adversary observes user connects to MOSAIC at 14:05
2. Protest begins at 14:00
3. Adversary infers: User is at protest
```

**Impact**: MEDIUM
- Temporal correlation reveals user's likely context

**Mitigation**:
```python
class TemporalObfuscation:
    def randomize_broadcast_timing(self, event_timestamp):
        """
        Decouple network transmission from actual event time
        """
        # Client-side delay (0-300 seconds random)
        client_delay = random.randint(0, 300)

        # Server-side additional jitter (0-300 seconds)
        server_jitter = random.randint(-300, 300)

        # Total temporal smearing: ±10 minutes
        transmitted_timestamp = event_timestamp + client_delay + server_jitter

        return transmitted_timestamp

    def scheduled_batch_uploads(self):
        """
        Batch events and upload at fixed intervals (e.g., every hour)
        """
        event_queue = []

        # Collect events over time
        event_queue.append(new_event)

        # Upload at fixed schedule (breaks timing correlation)
        if current_time() % 3600 == 0:  # Every hour
            self.upload_batch(event_queue)
            event_queue.clear()
```

**Residual Risk**: LOW
- ±10 minute timing uncertainty prevents precise correlation
- Scheduled batching breaks real-time correlation

---

#### 1.3 TLS Fingerprinting

**Attack**: Identify MOSAIC clients by TLS handshake characteristics

```
TLS fingerprint includes:
- Cipher suite order
- Extension list
- Supported curves
- TLS version
- SNI (Server Name Indication)
```

**Impact**: MEDIUM
- Unique TLS fingerprint identifies MOSAIC client

**Mitigation**:
```python
class TLSObfuscation:
    def randomize_tls_fingerprint(self):
        """
        Mimic common browsers to blend in
        """
        browser_profiles = [
            'Chrome/120.0',
            'Firefox/121.0',
            'Safari/17.0',
        ]

        # Select random browser profile
        profile = random.choice(browser_profiles)

        # Configure TLS to match browser
        tls_config = {
            'ciphers': get_browser_ciphers(profile),
            'curves': get_browser_curves(profile),
            'extensions': get_browser_extensions(profile),
        }

        return tls_config

    def domain_fronting(self):
        """
        Use CDN domain fronting (where available)
        """
        # SNI: cloudfront.com
        # Host header: mosaic-api.example.com
        # Appears as generic CDN traffic
```

**Residual Risk**: VERY LOW
- Randomized TLS fingerprints blend with normal traffic
- Domain fronting hides true destination

---

### 1.4 DNS Surveillance

**Attack**: Monitor DNS queries to detect MOSAIC usage

```
User queries: api.mosaic.example.com
→ Reveals MOSAIC usage
```

**Impact**: HIGH
- DNS queries reveal intent to use MOSAIC

**Mitigation**:
```python
class DNSPrivacy:
    def use_encrypted_dns(self):
        """
        Use DNS-over-HTTPS (DoH) or DNS-over-TLS (DoT)
        """
        doh_resolver = 'https://cloudflare-dns.com/dns-query'

        query = dns.query.https(
            'api.mosaic.example.com',
            where=doh_resolver
        )

        return query

    def use_tor_dns(self):
        """
        Route DNS through Tor (prevents resolver surveillance)
        """
        # Tor handles DNS resolution internally
        # No cleartext DNS queries
```

**Residual Risk**: VERY LOW
- Encrypted DNS prevents passive observation
- Tor eliminates DNS leakage entirely

---

## Threat Model 2: Compromised Server

### Adversary Profile

**Capability**: Level 2-3
- **Access**: Full control of coordination server
- **Can**: Read database, modify code, log traffic
- **Cannot**: Compromise client devices, break cryptography
- **Goal**: Deanonymize users, identify true signals in obfuscated data

### Attack Vectors

#### 2.1 Database Dump Analysis

**Attack**: Adversary dumps database, analyzes events offline

```
Database contains:
- Obfuscated events (60% client + 10% server noise = 70% total)
- Association requests (encrypted contacts)
- Session tokens (stateless, no user identity)
```

**Impact**: MEDIUM
- Adversary sees obfuscated events but cannot identify true signals
- Cannot link events to users (no identity in session tokens)

**Mitigation**:
```python
class ServerCompromiseResistance:
    def client_side_majority_obfuscation(self):
        """
        Client applies 60%+ noise before transmission
        Server compromise reveals ≤40% signal
        """
        assert client_noise_ratio >= 0.60

        # Even if server malicious, sees mostly noise
        # Cannot extract true signal without user collaboration

    def use_real_data_mixing(self, user_context):
        """
        Mix with other users' real context (not AI-generated)
        Adversary cannot distinguish true from noise
        """
        other_users_contexts = sample_recent_real_contexts(n=10)
        mixed = user_context + flatten(other_users_contexts)
        cryptographic_shuffle(mixed)

        return mixed  # All elements are real, indistinguishable
```

**Residual Risk**: LOW
- Compromised server learns ≤40% of true signal
- Real data mixing prevents filtering noise

---

#### 2.2 Logging Pre-Obfuscation Data

**Attack**: Malicious server logs data before applying server-side obfuscation

```
Scenario:
1. Client sends 60%-obfuscated event
2. Malicious server logs it
3. Server adds 10% noise (for appearances)
4. Server stores original 60%-obfuscated version
```

**Impact**: MEDIUM
- Server builds database of 60%-obfuscated events

**Mitigation**:
```python
class AntiLoggingProtection:
    def client_obfuscation_verification(self):
        """
        Client verifies server didn't log pre-obfuscation data
        """
        # Client includes commitment to obfuscated event
        event_hash = hash(obfuscated_event)
        commitment = commit(event_hash, random_nonce)

        # Server must return same hash in response
        # If server modified, hash won't match → client detects

    def federated_secret_sharing(self, event):
        """
        Split event across 3 providers (threshold=2)
        No single provider has full event
        """
        shares = shamir_split(event, threshold=2, total=3)

        send_to_provider_A(shares[0])  # AWS (US)
        send_to_provider_B(shares[1])  # OVH (France)
        send_to_provider_C(shares[2])  # Hetzner (Germany)

        # Requires collusion of 2+ providers to reconstruct
```

**Residual Risk**: VERY LOW (with federation)
- Single compromised provider sees only share (unusable)
- Client-side verification detects tampering

---

#### 2.3 Selective Event Deletion

**Attack**: Server deletes events selectively to fingerprint users

```
Scenario:
1. User broadcasts event
2. Server stores but doesn't make available in queries
3. User queries, doesn't see their own event
4. Server learns: This user was the broadcaster
```

**Impact**: HIGH
- Server can uniquely identify users by selective deletion

**Mitigation**:
```python
class SelectiveDeletionDefense:
    def probabilistic_querying(self):
        """
        Users don't expect to see their own events
        (Indistinguishable from normal obfuscation/filtering)
        """
        # Users broadcast but may not see own event in results
        # Normal: Event filtered by proximity/context mismatch
        # Attack: Server deleted it
        # User cannot distinguish → attack doesn't leak info

    def distributed_replication(self):
        """
        Events replicated across multiple nodes
        """
        # DHT: Event stored on K=20 nodes
        # Server deleting from one node doesn't prevent discovery
        # User queries multiple nodes in parallel

    def client_side_verification(self):
        """
        Users verify events are actually stored
        """
        # After broadcast, client queries for own event (different session)
        # If not found after multiple attempts → alert user
```

**Residual Risk**: LOW
- DHT replication prevents single-point deletion
- Probabilistic querying limits information leakage

---

#### 2.4 Injection of Malicious Events

**Attack**: Server injects fake events to fingerprint or mislead users

```
Scenario:
1. User queries proximity
2. Server injects fake event with unique marker
3. If user responds to fake event → server learns user's identity
```

**Impact**: MEDIUM
- Honeypot events can reveal user behavior

**Mitigation**:
```python
class InjectionDefense:
    def event_authenticity_verification(self):
        """
        Events are signed by client (even if anonymously)
        """
        # Client signs event with ephemeral key
        signature = ed25519_sign(ephemeral_private_key, event)

        # Server can verify signature but not link to user
        # Injected events lack valid signature → detectable

    def reputation_system(self):
        """
        Track event source reputation (anonymous but consistent)
        """
        # Each session has anonymous reputation score
        # New sessions start with low reputation
        # Injected events have low reputation → filtered by clients

    def client_side_filtering(self):
        """
        Clients apply own filtering/validation
        """
        # Sanity checks: plausible location, context, timing
        # Reject obviously fake events (e.g., impossible locations)
```

**Residual Risk**: LOW
- Signature verification prevents undetectable injection
- Client-side filtering catches anomalies

---

## Threat Model 3: Statistical Correlation Attack

### Adversary Profile

**Capability**: Level 3 (Nation-State)
- **Access**: Long-term database of events, advanced analytics
- **Can**: Machine learning, pattern matching, external data correlation
- **Cannot**: Break cryptography, compromise individual devices
- **Goal**: Deanonymize users through long-term pattern analysis

### Attack Vectors

#### 3.1 Behavioral Pattern Matching

**Attack**: Identify users by unique behavior patterns over time

```
Example:
User always broadcasts:
- Weekdays 9am-5pm near "Financial District"
- Weekends near "Residential Area X"
- Tuesdays also visits "Gym at 7pm"

Pattern is unique → user identifiable even with obfuscation
```

**Impact**: HIGH
- Long-term patterns overcome short-term obfuscation

**Mitigation**:
```python
class BehavioralObfuscation:
    def session_rotation_aggressive(self):
        """
        Rotate session every 30-60 minutes
        """
        # New session = new pseudonym
        # Adversary cannot link sessions over time

    def synthetic_pattern_injection(self):
        """
        Periodically broadcast synthetic events (chaff)
        """
        if random.random() < 0.2:  # 20% of time
            broadcast_synthetic_event(
                location=random_plausible_location(),
                context=random_plausible_context(),
                timing=random_time()
            )

        # Adds noise to user's behavioral pattern

    def differential_privacy_trajectory(self):
        """
        Add DP noise to location trajectory
        """
        # Not just individual locations, but trajectory patterns
        # ε-DP guarantee on sequence of locations
```

**Residual Risk**: MEDIUM
- Aggressive session rotation limits pattern accumulation
- Chaff traffic confuses pattern matching
- Still vulnerable if adversary has very long observation period (months+)

---

#### 3.2 External Data Correlation

**Attack**: Correlate MOSAIC events with external data sources

```
Example:
1. Social media post: "At protest downtown #SaveTheWhales 14:05"
2. MOSAIC event: Location=downtown, context="protest", time~14:05
3. Correlation: Link social media account → MOSAIC user
```

**Impact**: CRITICAL
- User's own OPSEC failure deanonymizes

**Mitigation**:
```python
class CorrelationResistance:
    def user_education(self):
        """
        Warn users about correlation risks
        """
        warnings = [
            "Don't post about MOSAIC usage on social media",
            "Don't broadcast events simultaneously with public posts",
            "Use MOSAIC from different network than social media",
            "Consider ±30 min timing offset for sensitive events"
        ]

    def automatic_sanitization(self):
        """
        Detect and warn about highly correlatable events
        """
        if event.context contains unique_identifiers:
            warn_user("This context may be correlatable with public data")
            suggest_alternatives()

    def timing_fuzzing_increased(self):
        """
        For high-risk events, increase temporal obfuscation
        """
        if user_selected_risk_level == "HIGH":
            temporal_jitter = random.randint(-1800, 1800)  # ±30 min
```

**Residual Risk**: HIGH
- User behavior is the weakest link
- System cannot prevent user mistakes
- Education and warnings reduce but don't eliminate risk

---

#### 3.3 Cross-Database Correlation

**Attack**: Combine MOSAIC data with other compromised databases

```
Example:
1. Adversary compromises Starbucks WiFi logs
2. Cross-reference: Same IP seen in Starbucks + MOSAIC event nearby
3. Correlation reveals: User X was at Starbucks at time T
```

**Impact**: HIGH
- Multiple data sources enable triangulation

**Mitigation**:
```python
class CrossDatabaseDefense:
    def network_isolation(self):
        """
        Encourage separate networks for MOSAIC vs. daily browsing
        """
        # Use mobile data for MOSAIC
        # Use WiFi for general browsing
        # Prevents IP correlation

    def tor_mandatory_mode(self):
        """
        Optional mode: All MOSAIC traffic through Tor
        """
        # IP addresses completely decoupled from location
        # Exit node IP != user's real IP

    def mac_address_randomization(self):
        """
        Randomize MAC address when connecting to networks
        """
        # iOS/Android support random MAC per network
        # Prevents device fingerprinting via WiFi
```

**Residual Risk**: MEDIUM
- Network isolation significantly reduces correlation risk
- Tor provides strong IP decorrelation
- Still vulnerable if user uses same device/browser fingerprint

---

#### 3.4 Intersection Attacks

**Attack**: Identify user by who was in multiple locations

```
Example:
3 events within 1 hour:
- Event A: Location X, 10 candidate users
- Event B: Location Y, 8 candidate users
- Event C: Location Z, 12 candidate users

Intersection: Only 1 user present at all 3 locations → identified
```

**Impact**: CRITICAL
- Multiple observations rapidly narrow candidate set

**Mitigation**:
```python
class IntersectionAttackDefense:
    def k_anonymity_enforcement(self):
        """
        Ensure each event has ≥k plausible sources
        """
        MIN_K = 10

        event_candidates = query_users_in_proximity(event.location, event.time)

        if len(event_candidates) < MIN_K:
            # Inject synthetic events from other locations
            # Or delay event until more users nearby
            inject_synthetic_user_events(count=MIN_K - len(event_candidates))

    def synthetic_trajectory_injection(self):
        """
        System generates synthetic user trajectories
        """
        # Fake users with realistic movement patterns
        # Pollute intersection analysis with decoys

    def broadcast_suppression_sparse_areas(self):
        """
        Warn users when broadcasting from sparse area
        """
        if estimated_nearby_users < 10:
            warn_user("Low anonymity set! Consider waiting or moving.")
```

**Residual Risk**: MEDIUM
- K-anonymity helps but not perfect (adversary can still narrow set)
- Synthetic users add noise but detectable with advanced ML
- Best defense: User education about sparse-area risks

---

## Threat Model 4: Compelled Client Modification

### Adversary Profile

**Capability**: Level 3 (State Actor)
- **Access**: Legal power to compel user to install modified client
- **Can**: Force app update, demand unlock codes, physical device access
- **Cannot**: Break cryptography, access secure enclave without biometrics
- **Goal**: Extract pre-obfuscation data from client before transmission

### Attack Vectors

#### 4.1 Forced App Backdoor

**Attack**: Court order to user: "Install this modified MOSAIC app"

```
Modified app:
- Logs true location before obfuscation
- Logs true context before noise injection
- Sends logs to law enforcement
```

**Impact**: CRITICAL
- Compromised client reveals all user activity

**Mitigation**:
```python
class AntiBackdoorProtection:
    def code_signing_verification(self):
        """
        App verifies its own binary integrity on launch
        """
        app_hash = hash_executable_code()
        official_hash = fetch_from_trusted_source("https://mosaic.org/hash")

        if app_hash != official_hash:
            alert_user("App may be modified! Don't use.")
            exit()

    def reproducible_builds(self):
        """
        Anyone can verify official app matches source code
        """
        # Deterministic compilation
        # Hash matches published source code
        # User can compile themselves if paranoid

    def dead_mans_switch(self):
        """
        If device offline >7 days, auto-delete keys
        """
        # Prevents compelled use of stale compromised device
        # Adversary must compel in real-time (harder)

    def plausible_deniability_mode(self):
        """
        App generates plausible fake activity if compelled
        """
        # Hidden "duress mode" password
        # Generates realistic but fake events
        # User can claim: "This is my real activity"
```

**Residual Risk**: MEDIUM
- Code signing detects modification
- Reproducible builds enable community verification
- Dead man's switch limits compromise window
- Still vulnerable if user coerced in real-time

---

#### 4.2 Device Seizure & Forensics

**Attack**: Law enforcement seizes device, extracts data forensically

```
Targets:
- App storage (cached events, contacts)
- Keychain (session keys, passwords)
- Memory dumps (active encryption keys)
```

**Impact**: HIGH
- Offline device analysis can reveal sensitive data

**Mitigation**:
```python
class ForensicsResistance:
    def full_disk_encryption(self):
        """
        Require strong device password
        """
        # iOS/Android FDE standard
        # Without password, storage encrypted

    def secure_enclave_storage(self):
        """
        Store keys in hardware-protected enclave
        """
        # iOS: Keychain with kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        # Android: Keystore with setUserAuthenticationRequired(true)

        # Keys inaccessible without biometric/PIN

    def ephemeral_memory_only(self):
        """
        Don't persist sensitive data to disk
        """
        # Session keys: RAM only
        # Contacts: RAM only (delete after association)
        # Events: Encrypted before storage, keys in secure enclave

    def anti_forensics_wipe(self):
        """
        Wipe on tampering detection
        """
        # Detect: Jailbreak, debugger, memory dump tools
        # Response: Zeroize keys immediately
```

**Residual Risk**: LOW
- FDE + secure enclave prevents offline extraction
- Ephemeral data minimizes attack surface
- Anti-forensics makes memory dumps difficult

---

#### 4.3 Man-in-the-Middle Forced Update

**Attack**: ISP/government injects fake app update

```
Scenario:
1. User opens app, checks for update
2. Attacker intercepts update check
3. Serves malicious "update" signed with stolen or compelled cert
```

**Impact**: CRITICAL
- Malicious update installs backdoor

**Mitigation**:
```python
class UpdateSecurity:
    def certificate_pinning(self):
        """
        App only trusts specific certificates
        """
        trusted_certs = [
            "sha256/ABC123...",  # Primary cert
            "sha256/DEF456...",  # Backup cert
        ]

        if server_cert not in trusted_certs:
            reject_update()

    def multi_signature_requirement(self):
        """
        Updates require signatures from 3+ developers
        """
        # No single developer can push malicious update
        # Adversary must compromise 3+ keys

    def transparency_log(self):
        """
        All updates published to append-only log
        """
        # Certificate Transparency style
        # Community can audit: "This update not in log!"

    def manual_update_option(self):
        """
        Users can download updates manually
        """
        # Paranoid users can verify hash themselves
        # Download from mirror, compile from source
```

**Residual Risk**: LOW
- Certificate pinning prevents MITM
- Multi-sig prevents single-point compromise
- Transparency log enables community detection

---

## Threat Model 5: Social Graph Reconstruction

### Adversary Profile

**Capability**: Level 2-3
- **Access**: Long-term observation of all events
- **Can**: Pattern matching, graph analysis, temporal correlation
- **Cannot**: Decrypt content, compromise devices
- **Goal**: Map who knows whom (social graph)

### Attack Vectors

#### 5.1 Repeated Proximity Detection

**Attack**: Two pseudonyms repeatedly seen in proximity → likely know each other

```
Example:
Session A and Session B appear together:
- Monday 14:00, Location X
- Wednesday 12:00, Location Y
- Friday 18:00, Location Z

Probability they know each other: HIGH
```

**Impact**: HIGH
- Social relationships revealed even with session rotation

**Mitigation**:
```python
class SocialGraphObfuscation:
    def session_rotation_coordination(self):
        """
        Users rotate sessions at different times
        """
        # Alice rotates every 30 min
        # Bob rotates every 45 min
        # Prevents long-term session co-occurrence

    def synthetic_proximity_injection(self):
        """
        System injects fake co-location events
        """
        # Random pairs of users "appear" together
        # False proximity pollutes graph analysis

    def privacy_preserving_group_coordination(self):
        """
        For group events, use shared token instead of proximity
        """
        # Organizer generates token: "protest_xyz_123"
        # Participants broadcast token (not location)
        # System doesn't learn who was physically together

    def differential_privacy_social_graph(self):
        """
        Add DP noise to graph edges
        """
        # Randomly flip edges (add/remove connections)
        # ε-DP guarantee on graph structure
```

**Residual Risk**: MEDIUM
- Synthetic injection adds noise
- Shared tokens break location-based correlation
- Still vulnerable to very long-term observation (months)

---

#### 5.2 Association Pattern Analysis

**Attack**: Analyze association requests to infer relationships

```
Example:
Session A → Association request for event from Session B (10 times)
→ A and B likely know each other
```

**Impact**: MEDIUM
- Association patterns leak social graph

**Mitigation**:
```python
class AssociationPatternDefense:
    def association_tokens_single_use(self):
        """
        Each token used exactly once
        """
        # Cannot correlate multiple associations via token reuse

    def blind_association_requests(self):
        """
        Server doesn't learn who requests association with whom
        """
        # Use blind signature protocol
        # Server facilitates without learning endpoints

    def decoy_association_requests(self):
        """
        Users periodically request fake associations
        """
        if random.random() < 0.1:  # 10% of time
            request_association(random_event_token)
            # Never follow through with actual contact

        # Pollutes association graph with false edges
```

**Residual Risk**: LOW
- Blind signatures hide association patterns from server
- Decoy requests add noise

---

## Threat Model 6: Adversarial Machine Learning

### Adversary Profile

**Capability**: Level 2-3
- **Access**: MOSAIC database, advanced ML models
- **Can**: Train classifiers, GANs, anomaly detectors
- **Cannot**: Compromise cryptography or devices
- **Goal**: Distinguish real signals from synthetic noise

### Attack Vectors

#### 6.1 Synthetic Context Detection

**Attack**: ML classifier trained to identify AI-generated vs. real context

```python
class DeobfuscationClassifier:
    def train(self, real_contexts, synthetic_contexts):
        features = extract_features(real_contexts + synthetic_contexts)
        labels = [1] * len(real_contexts) + [0] * len(synthetic_contexts)

        self.model = GradientBoostingClassifier()
        self.model.fit(features, labels)

    def predict_real_probability(self, context_element):
        features = extract_features([context_element])
        return self.model.predict_proba(features)[0][1]

    def filter_noise(self, obfuscated_event):
        # Classify each element
        probabilities = [
            self.predict_real_probability(elem)
            for elem in obfuscated_event.context
        ]

        # Keep only elements with >80% real probability
        real_elements = [
            elem for elem, prob in zip(obfuscated_event.context, probabilities)
            if prob > 0.8
        ]

        return real_elements  # Deobfuscated!
```

**Impact**: CRITICAL
- If successful, adversary removes 70% noise → sees true signal

**Mitigation**:
```python
class AdversarialMLDefense:
    def use_real_data_mixing(self, user_context):
        """
        BEST DEFENSE: Mix with other users' REAL context
        """
        # All elements are real → classifier useless
        other_users = sample_recent_real_contexts(n=10)
        mixed = user_context + flatten(other_users)
        shuffle(mixed)

        return mixed  # 100% real data, indistinguishable

    def adversarial_training(self):
        """
        Train noise generator to fool classifiers
        """
        generator = ContextGenerator()
        discriminator = DeobfuscationClassifier()

        for epoch in range(1000):
            # Generate synthetic contexts
            synthetic = generator.generate(real_contexts)

            # Train discriminator
            discriminator.train(real_contexts, synthetic)

            # Update generator to fool discriminator
            if discriminator.accuracy > 0.55:  # Too good
                generator.adversarial_update(discriminator)

        # Target: Discriminator accuracy = 50% (random guessing)

    def multimodal_obfuscation(self):
        """
        Harder for ML to classify multimodal data
        """
        # Mix text, images, audio, location
        # Classifier must handle all modalities → harder
```

**Residual Risk**: VERY LOW (with real data mixing)
- Real data mixing makes detection impossible
- Adversarial training ensures synthetic is indistinguishable
- Multimodal complexity increases adversary cost

---

## Threat Model 7: Collusion Attacks

### Adversary Profile

**Capability**: Level 3
- **Access**: Multiple compromised components (clients, servers, providers)
- **Can**: Coordinate attacks across components
- **Cannot**: Break fundamental cryptography
- **Goal**: Overcome security through collusion

### Attack Vectors

#### 7.1 Multi-Provider Collusion

**Attack**: 2+ federated providers collude to reconstruct events

```
Federated setup:
- Provider A (AWS, US)
- Provider B (OVH, France)
- Provider C (Hetzner, Germany)
- Threshold = 2

If A + B collude:
- Combine shares → reconstruct events
- See client-obfuscated data (60% noise)
```

**Impact**: MEDIUM
- Collusion reveals 40% signal (better than 0% but not full deanonymization)

**Mitigation**:
```python
class CollusionResistance:
    def choose_adversarial_providers(self):
        """
        Select providers with conflicting interests
        """
        providers = [
            "AWS (US - Five Eyes)",
            "OVH (France - EU Privacy Laws)",
            "Hetzner (Germany - Strong Privacy)",
            "Yandex (Russia - Anti-US)",
            "Alibaba (China - Different jurisdiction)"
        ]

        # Geopolitical/legal barriers to collusion

    def increase_threshold(self):
        """
        Require more providers to reconstruct
        """
        # threshold=3, total=5 providers
        # Need 3+ to collude (harder)

    def client_obfuscation_majority(self):
        """
        Even with collusion, only see 40% signal
        """
        # 60% client noise is unrecoverable
        # Federated providers see pre-server-noise data
```

**Residual Risk**: LOW
- Geopolitical diversity makes collusion difficult
- Higher thresholds exponentially increase collusion cost
- Client obfuscation limits damage even if collusion succeeds

---

## Threat Model 8: Side-Channel Attacks

### Adversary Profile

**Capability**: Level 3 (Advanced)
- **Access**: Physical proximity, special equipment
- **Can**: Measure power, EM radiation, timing
- **Cannot**: Break cryptography directly
- **Goal**: Extract secrets via side channels

### Attack Vectors

#### 8.1 Timing Attacks

**Attack**: Measure timing variations to infer secret values

```
Example:
- If password check: for c in password: if c != input[i]: return False
- Timing reveals password length and partial contents
```

**Impact**: HIGH (if not defended)
- Can extract keys, passwords, etc.

**Mitigation**:
```python
class TimingAttackDefense:
    def constant_time_comparison(self, a, b):
        """
        All comparisons take same time regardless of values
        """
        if len(a) != len(b):
            # Still constant time (compare dummy values)
            diff = 0
            for x, y in zip(a, [0]*len(a)):
                diff |= x ^ y
            return False

        diff = 0
        for x, y in zip(a, b):
            diff |= x ^ y

        return diff == 0  # Single branch at end

    def randomize_processing_time(self):
        """
        Add random delays to mask timing variations
        """
        processing_time = actual_work()
        random_delay = random.uniform(0, 0.1)  # 0-100ms
        sleep(random_delay)

        return result
```

**Residual Risk**: VERY LOW
- Constant-time crypto libraries standard (libsodium, etc.)
- Random delays mask any residual leakage

---

#### 8.2 Power Analysis

**Attack**: Measure device power consumption to extract keys

```
Example:
- AES encryption power consumption varies by key bits
- Differential Power Analysis (DPA) can extract keys
```

**Impact**: CRITICAL (if physical access)
- Can extract encryption keys from device

**Mitigation**:
```python
class PowerAnalysisDefense:
    def use_hardware_crypto(self):
        """
        Delegate to secure hardware (Secure Enclave, TPM)
        """
        # Hardware designed with power analysis resistance
        # Constant power draw, randomized operations

    def software_masking(self):
        """
        Add random masking to computations
        """
        # Mask = random value
        # Compute on (value XOR mask)
        # Unmask result
        # Power consumption randomized

    def physical_security(self):
        """
        Require physical access prevention
        """
        # Users should not leave devices unattended
        # Enable "wipe on tamper" if available
```

**Residual Risk**: LOW
- Modern secure enclaves resistant to power analysis
- Physical access required (high barrier)

---

## Threat Model 9: Legal Compulsion

### Adversary Profile

**Capability**: Level 3 (State Actor)
- **Access**: Legal authority to demand data/access
- **Can**: Subpoena, NSL, court order, gag order
- **Cannot**: Break cryptography, force users (in free societies)
- **Goal**: Compel disclosure of user data or system backdoors

### Attack Vectors

#### 9.1 Server Data Subpoena

**Attack**: Government orders MOSAIC to hand over all data

```
Subpoena:
"Provide all data related to user X between dates A and B"
```

**Impact**: MEDIUM (limited by design)
- Server has no user identities to link
- Events are obfuscated
- Association records auto-deleted

**Mitigation**:
```python
class LegalCompulsionResistance:
    def data_minimization(self):
        """
        Don't store what you can't be compelled to provide
        """
        no_user_accounts = True
        no_persistent_identifiers = True
        no_payment_info = True  # Free service or crypto payments

    def time_to_live_enforcement(self):
        """
        Automatic data deletion
        """
        events_ttl = 48  # hours
        associations_ttl = 48  # hours

        # By time subpoena arrives, data already deleted

    def transparency_report(self):
        """
        Publish all legal requests (when legally allowed)
        """
        quarterly_report = {
            "subpoenas_received": 0,
            "data_provided": 0,
            "warrant_canary": "present"
        }
```

**Residual Risk**: LOW
- Minimal data retention limits compliance
- Obfuscation limits value of provided data

---

#### 9.2 Forced Backdoor

**Attack**: Secret court order to implement backdoor

```
NSL/FISA: "Implement logging of all user activity, don't tell anyone"
```

**Impact**: CRITICAL
- Backdoor undermines all privacy

**Mitigation**:
```python
class BackdoorResistance:
    def warrant_canary(self):
        """
        Daily statement: "We have not received secret orders"
        """
        canary_text = f"""
        As of {today()}, MOSAIC has:
        - Not received National Security Letters
        - Not received FISA orders
        - Not received gag orders
        - Not implemented backdoors
        """

        # If canary disappears → community knows

    def offshore_incorporation(self):
        """
        Incorporate in jurisdiction resistant to US/EU compulsion
        """
        jurisdictions = [
            "Switzerland",  # Strong privacy laws
            "Iceland",      # Journalist protections
            "Seychelles"    # Offshore, minimal cooperation
        ]

    def open_source_all_code(self):
        """
        Community can audit for backdoors
        """
        # Reproducible builds
        # Any backdoor would be visible in source

    def decentralize_before_compulsion(self):
        """
        If compulsion threatened, accelerate decentralization
        """
        # Move to DHT/P2P
        # No company to compel
```

**Residual Risk**: MEDIUM
- Warrant canary provides warning
- Open source enables detection
- Decentralization eliminates target

---

## Threat Model 10: Sybil Attacks

### Adversary Profile

**Capability**: Level 2
- **Access**: Can create many fake identities cheaply
- **Can**: Flood system with fake users/events
- **Cannot**: Compromise crypto or real users' devices
- **Goal**: Disrupt service, manipulate data, surveil real users

### Attack Vectors

#### 10.1 Event Flooding

**Attack**: Create thousands of fake events to overwhelm system

```
Attacker:
for i in range(100000):
    broadcast_event(fake_location, fake_context)

Result: Real events drowned in fake events
```

**Impact**: HIGH
- Denial of service
- Real users cannot find legitimate events

**Mitigation**:
```python
class SybilDefense:
    def proof_of_work_broadcast(self):
        """
        Require computational cost to broadcast
        """
        difficulty = 4  # 4 leading zero bits (2^4 = 16 attempts avg)

        challenge = hash(event + server_nonce)
        nonce = find_nonce_where(hash(challenge + nonce).startswith("0000"))

        # Client must prove work before broadcast accepted

        # Cost for attacker: 16 × 100,000 = 1.6M hashes
        # Time: ~10 seconds per event on mobile
        # Makes mass flooding expensive

    def rate_limiting_aggressive(self):
        """
        Limit broadcasts per IP/session
        """
        limits = {
            "broadcasts_per_hour": 10,
            "queries_per_hour": 100,
            "associations_per_day": 5
        }

    def reputation_based_filtering(self):
        """
        New sessions have low reputation
        """
        # Reputation increases with age + legitimate usage
        # Low reputation events filtered by clients

        if event.session_reputation < 0.5:
            filter_out()

    def economic_cost(self):
        """
        Charge small fee per broadcast (crypto microtransaction)
        """
        # $0.01 per broadcast (Lightning Network)
        # 100,000 events = $1,000 (expensive attack)
```

**Residual Risk**: LOW
- Proof of work makes flooding expensive
- Rate limiting contains damage
- Reputation + economic cost deter attacks

---

## Attack Trees

### Attack Tree: Deanonymize User

```
Goal: Identify real-world identity of MOSAIC user

OR
├── Compromise Client Device
│   AND
│   ├── Physical Access [HIGH DIFFICULTY]
│   ├── Bypass Encryption [HIGH DIFFICULTY]
│   └── Extract Keys [HIGH DIFFICULTY]
│
├── Long-Term Pattern Analysis
│   AND
│   ├── Collect Events (months) [LOW DIFFICULTY]
│   ├── Correlate Patterns [MEDIUM DIFFICULTY]
│   └── Match External Data [MEDIUM DIFFICULTY]
│
├── Compromise Server + ML Deobfuscation
│   AND
│   ├── Compromise Server [MEDIUM DIFFICULTY]
│   ├── Train Classifier [MEDIUM DIFFICULTY]
│   └── Filter Noise [HIGH DIFFICULTY - mitigated by real data mixing]
│
└── Legal Compulsion
    AND
    ├── Court Order [MEDIUM DIFFICULTY]
    └── User Compliance [VARIABLE - depends on jurisdiction/user]
```

**Hardest Path**: All require either long-term observation + correlation, or physical device access

---

## Risk Matrix

| Threat Model | Likelihood | Impact | Residual Risk | Priority |
|--------------|------------|--------|---------------|----------|
| Passive Network Observer | HIGH | MEDIUM | LOW | HIGH (Common threat) |
| Compromised Server | MEDIUM | MEDIUM | LOW | HIGH (Critical defense) |
| Statistical Correlation | MEDIUM | HIGH | MEDIUM | CRITICAL |
| Compelled Client Mod | LOW | CRITICAL | MEDIUM | MEDIUM |
| Social Graph Recon | MEDIUM | HIGH | MEDIUM | HIGH |
| Adversarial ML | MEDIUM | CRITICAL | LOW | CRITICAL |
| Collusion Attack | LOW | MEDIUM | LOW | LOW |
| Side-Channel | LOW | HIGH | LOW | LOW |
| Legal Compulsion | MEDIUM | HIGH | LOW | HIGH |
| Sybil Attack | MEDIUM | MEDIUM | LOW | MEDIUM |

**Highest Priority Mitigations**:
1. Statistical correlation defense (session rotation, chaff, DP)
2. Adversarial ML defense (real data mixing)
3. Passive observer defense (Tor, traffic obfuscation)
4. Compromised server defense (client-majority obfuscation, federation)

---

## Conclusion

MOSAIC's threat model is comprehensive and addresses realistic adversaries from individual attackers to nation-states. The multi-layered defense strategy provides:

1. **Information-theoretic deniability** through noise injection
2. **Computational deniability** through cryptography
3. **Practical deniability** through legal/jurisdictional positioning

**Key Insight**: Perfect security is impossible, but MOSAIC increases adversary cost from negligible to >$100,000+ per user deanonymization, putting it out of reach for all but the most resourced and motivated attackers.

**Weakest Points**:
- Long-term behavioral pattern analysis (months+ observation)
- User OPSEC failures (correlation with public data)
- Compelled client modification in authoritarian regimes

**Strongest Points**:
- Client-majority obfuscation (server compromise reveals ≤40% signal)
- Real data mixing (adversarial ML becomes impossible)
- Federated architecture (no single point of compromise)
- Ephemeral identity (no long-term tracking)

---

**Document Version**: 1.0.0
**Last Security Review**: 2025-11-09
**Next Review**: Quarterly
**Status**: Threat Model Complete - Inform Design Decisions
