/******************************************************************************
 *   ███████╗██╗     ██╗████████╗███████╗    ██╗   ██╗███████╗███████╗██████╗
 *   ██╔════╝██║     ██║╚══██╔══╝██╔════╝    ██║   ██║██╔════╝██╔════╝██╔══██╗
 *   █████╗  ██║     ██║   ██║   █████╗      ██║   ██║███████╗█████╗  ██████╔╝
 *   ██╔══╝  ██║     ██║   ██║   ██╔══╝      ██║   ██║╚════██║██╔══╝  ██╔══██╗
 *   ███████╗███████╗██║   ██║   ███████╗    ╚██████╔╝███████║███████╗██║  ██║
 *   ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝     ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝
 *                            .  ╿  S
 * 
 *  ⚠️  WARNING: This config WILL break many websites. That's the point.
 *  ⚠️  Use a separate profile for general browsing if needed.
 *  ⚠️  Review each section - some prefs may conflict with your workflow.
 * 
 *  RECOMMENDED EXTENSIONS:
 *    • uBlock Origin (hard mode)
 *    • NoScript
 *    • CanvasBlocker
 *    • Skip Redirect
 *    • LocalCDN/Decentraleyes
 *    • ClearURLs
 *    • Temporary Containers
 *    • Multi-Account Containers
 * 
 *  DEPLOYMENT:
 *    1. Create new profile: firefox -P
 *    2. Navigate to profile folder: about:profiles
 *    3. Drop this file as user.js
 *    4. Restart Firefox
 *    5. Verify with about:config
 ******************************************************************************/


/******************************************************************************
 * SECTION 0: STARTUP & META CONFIGURATION
 * Disable first-run annoyances, update checks, and telemetry handshakes
 ******************************************************************************/

// [CRITICAL] Disable automatic connections on startup
user_pref("network.connectivity-service.enabled", false);
user_pref("network.captive-portal-service.enabled", false);
user_pref("browser.selfsupport.url", "");

// Full URL display - never truncate
user_pref("browser.urlbar.trimURLs", false);
user_pref("browser.urlbar.trimHttps", false);

// Disable about:config warning
user_pref("browser.aboutConfig.showWarning", false);

// Disable default browser check
user_pref("browser.shell.checkDefaultBrowser", false);

// Disable What's New toolbar icon
user_pref("browser.messaging-system.whatsNewPanel.enabled", false);

// Disable extension recommendations
user_pref("browser.newtabpage.activity-stream.asrouter.userprefs.cfr.addons", false);
user_pref("browser.newtabpage.activity-stream.asrouter.userprefs.cfr.features", false);

// Disable startup homepage override
user_pref("browser.startup.homepage_override.mstone", "ignore");

// Disable UI tour
user_pref("browser.uitour.enabled", false);


/******************************************************************************
 * SECTION 1: ENHANCED TRACKING PROTECTION (ETP)
 * Firefox's built-in tracking protection - set to maximum
 ******************************************************************************/

// Enable all tracking protection categories
user_pref("privacy.trackingprotection.enabled", true);
user_pref("privacy.trackingprotection.pbmode.enabled", true);
user_pref("privacy.trackingprotection.socialtracking.enabled", true);
user_pref("privacy.trackingprotection.cryptomining.enabled", true);
user_pref("privacy.trackingprotection.fingerprinting.enabled", true);
user_pref("privacy.trackingprotection.emailtracking.enabled", true);

// Use strictest blocklist
user_pref("urlclassifier.trackingTable", "test-track-simple,base-track-digest256,content-track-digest256");

// Enable ETP strict mode
user_pref("browser.contentblocking.category", "strict");

// Block tracker referrers
user_pref("privacy.trackingprotection.annotate_channels", true);

// Block known fingerprinters
user_pref("privacy.fingerprintingProtection", true);


/******************************************************************************
 * SECTION 2: ADVANCED FINGERPRINTING RESISTANCE
 * Make your browser blend into the crowd
 ******************************************************************************/

// [CRITICAL] Master fingerprinting resistance switch
user_pref("privacy.resistFingerprinting", true);

// Letterboxing - obscures true viewport dimensions
user_pref("privacy.resistFingerprinting.letterboxing", true);
user_pref("privacy.resistFingerprinting.letterboxing.dimensions", "");

// Spoof EN-US locale regardless of system locale
user_pref("privacy.spoof_english", 2);

// Disable font enumeration (major fingerprint vector)
user_pref("browser.display.use_document_fonts", 0);
user_pref("layout.css.font-visibility.private", 1);
user_pref("layout.css.font-visibility.standard", 1);
user_pref("layout.css.font-visibility.trackingprotection", 1);

// Canvas fingerprinting protection
user_pref("privacy.resistFingerprinting.randomization.daily_reset.enabled", true);
user_pref("privacy.resistFingerprinting.randomization.daily_reset.private.enabled", true);

// Disable DOM gamepad (fingerprinting + not needed for most)
user_pref("dom.gamepad.enabled", false);
user_pref("dom.gamepad.extensions.enabled", false);
user_pref("dom.gamepad.extensions.lightindicator", false);
user_pref("dom.gamepad.extensions.multitouch", false);

// Disable battery status API (fingerprinting)
user_pref("dom.battery.enabled", false);

// Disable device sensors (accelerometer, gyroscope, etc.)
user_pref("device.sensors.enabled", false);
user_pref("device.sensors.motion.enabled", false);
user_pref("device.sensors.orientation.enabled", false);
user_pref("device.sensors.ambientLight.enabled", false);
user_pref("device.sensors.proximity.enabled", false);

// Disable vibrator API
user_pref("dom.vibrator.enabled", false);

// Hardware concurrency spoofing (CPU cores)
user_pref("dom.maxHardwareConcurrency", 2);

// Limit device memory reporting
user_pref("dom.navigator.maxDeviceMemory", 2);

// Disable screen orientation API
user_pref("dom.screenorientation.enabled", false);

// Disable VR/XR APIs (fingerprinting)
user_pref("dom.vr.enabled", false);
user_pref("dom.vr.oculus.enabled", false);
user_pref("dom.vr.openvr.enabled", false);
user_pref("dom.vr.osvr.enabled", false);


/******************************************************************************
 * SECTION 3: FIRST-PARTY ISOLATION & STATE PARTITIONING
 * Prevent cross-site tracking via storage mechanisms
 ******************************************************************************/

// [CRITICAL] First-party isolation - all storage keyed to first-party
// NOTE: May break cross-domain logins (SSO, OAuth flows)
user_pref("privacy.firstparty.isolate", true);
user_pref("privacy.firstparty.isolate.restrict_opener_access", true);

// Dynamic First-Party Isolation (dFPI) - successor to strict FPI
user_pref("network.cookie.cookieBehavior", 5);

// State partitioning (Total Cookie Protection)
user_pref("privacy.partition.network_state", true);
user_pref("privacy.partition.serviceWorkers", true);
user_pref("privacy.partition.always_partition_third_party_non_cookie_storage", true);
user_pref("privacy.partition.always_partition_third_party_non_cookie_storage.exempt_sessionstorage", false);
user_pref("privacy.partition.bloburl_per_partition_key", true);

// Partition network state
user_pref("privacy.partition.network_state.ocsp_cache", true);
user_pref("privacy.partition.network_state.ocsp_cache.pbmode", true);

// HSTS tracking prevention
user_pref("privacy.partition.network_state.hsts_telemetry.enabled", false);


/******************************************************************************
 * SECTION 4: COOKIES, CACHE & STORAGE HARDENING
 * Aggressive data minimization - session-only by default
 ******************************************************************************/

// Session-only cookies (cleared on shutdown)
user_pref("network.cookie.lifetimePolicy", 2);

// Third-party cookie handling (block all)
user_pref("network.cookie.thirdparty.sessionOnly", true);
user_pref("network.cookie.thirdparty.nonsecureSessionOnly", true);

// Disable cookie prefetching
user_pref("network.cookie.prefetch", false);

// [CRITICAL] Clear everything on shutdown
user_pref("privacy.sanitize.sanitizeOnShutdown", true);
user_pref("privacy.clearOnShutdown.cookies", true);
user_pref("privacy.clearOnShutdown.cache", true);
user_pref("privacy.clearOnShutdown.history", true);
user_pref("privacy.clearOnShutdown.formdata", true);
user_pref("privacy.clearOnShutdown.offlineApps", true);
user_pref("privacy.clearOnShutdown.sessions", true);
user_pref("privacy.clearOnShutdown.downloads", true);
user_pref("privacy.clearOnShutdown.siteSettings", false); // Keep permissions

// Firefox 128+ clear on shutdown options
user_pref("privacy.clearOnShutdown_v2.cookiesAndStorage", true);
user_pref("privacy.clearOnShutdown_v2.cache", true);
user_pref("privacy.clearOnShutdown_v2.historyFormDataAndDownloads", true);

// Disable history entirely
user_pref("places.history.enabled", false);
user_pref("browser.formfill.enable", false);

// Private browsing mode by default
user_pref("browser.privatebrowsing.autostart", true);

// Disable offline/application cache
user_pref("browser.cache.offline.enable", false);
user_pref("browser.cache.disk.enable", false);
user_pref("browser.cache.memory.capacity", 65536);

// IndexedDB restrictions
user_pref("dom.indexedDB.logging.enabled", false);

// Disable service workers (persistent background scripts)
user_pref("dom.serviceWorkers.enabled", false);

// Disable push notifications (requires service workers)
user_pref("dom.push.enabled", false);
user_pref("dom.push.connection.enabled", false);
user_pref("dom.push.serverURL", "");


/******************************************************************************
 * SECTION 5: NETWORK SECURITY & LEAK PREVENTION
 * HTTPS enforcement, DNS hardening, connection security
 ******************************************************************************/

// [CRITICAL] HTTPS-Only Mode
user_pref("dom.security.https_only_mode", true);
user_pref("dom.security.https_only_mode_ever_enabled", true);
user_pref("dom.security.https_only_mode_send_http_background_request", false);
user_pref("dom.security.https_only_mode.upgrade_local", true);
user_pref("dom.security.https_first", true);
user_pref("dom.security.https_first_pbm", true);

// HSTS preload enforcement
user_pref("network.stricttransportsecurity.preloadlist", true);

// [CRITICAL] Disable all prefetching/preconnection (prevents tracking + leaks)
user_pref("network.dns.disablePrefetch", true);
user_pref("network.dns.disablePrefetchFromHTTPS", true);
user_pref("network.prefetch-next", false);
user_pref("network.predictor.enabled", false);
user_pref("network.predictor.enable-prefetch", false);
user_pref("network.predictor.enable-hover-on-ssl", false);
user_pref("browser.urlbar.speculativeConnect.enabled", false);
user_pref("browser.places.speculativeConnect.enabled", false);

// Disable link prerendering
user_pref("network.preload", false);

// Disable speculative parallel connections
user_pref("network.http.speculative-parallel-limit", 0);

// [CRITICAL] Encrypted DNS (DNS-over-HTTPS)
// Mode: 0=off, 2=first, 3=only (strict)
user_pref("network.trr.mode", 3);
user_pref("network.trr.uri", "https://mozilla.cloudflare-dns.com/dns-query");
user_pref("network.trr.bootstrapAddress", "1.1.1.1");
user_pref("network.trr.early-AAAA", true);
user_pref("network.trr.wait-for-portal", false);

// Alternative DoH providers (uncomment to use):
// Quad9:
// user_pref("network.trr.uri", "https://dns.quad9.net/dns-query");
// user_pref("network.trr.bootstrapAddress", "9.9.9.9");
// Mullvad:
// user_pref("network.trr.uri", "https://doh.mullvad.net/dns-query");
// user_pref("network.trr.bootstrapAddress", "194.242.2.2");

// Disable system DNS fallback (all DNS through DoH)
user_pref("network.trr.allow-rfc1918", false);

// Encrypted Client Hello (ECH) - hide SNI
user_pref("network.dns.echconfig.enabled", true);
user_pref("network.dns.http3_echconfig.enabled", true);
user_pref("network.dns.use_https_rr_as_altsvc", true);

// [CRITICAL] Disable WebRTC (IP leak prevention behind VPN/Tor)
user_pref("media.peerconnection.enabled", false);
user_pref("media.peerconnection.ice.default_address_only", true);
user_pref("media.peerconnection.ice.no_host", true);
user_pref("media.peerconnection.ice.proxy_only_if_behind_proxy", true);
user_pref("media.peerconnection.use_document_iceservers", false);
user_pref("media.peerconnection.identity.timeout", 1);

// Disable WebRTC-related leaks
user_pref("media.navigator.enabled", false);
user_pref("media.navigator.video.enabled", false);

// Proxy settings (if using Tor/VPN SOCKS proxy)
// user_pref("network.proxy.type", 1);
// user_pref("network.proxy.socks", "127.0.0.1");
// user_pref("network.proxy.socks_port", 9050);
// user_pref("network.proxy.socks_remote_dns", true);
// user_pref("network.proxy.socks_version", 5);

// Prevent proxy bypass
user_pref("network.proxy.failover_direct", false);
user_pref("network.proxy.allow_bypass", false);

// IPv6 - disable if not needed (can leak)
user_pref("network.dns.disableIPv6", true);

// GIO protocol handler (Linux)
user_pref("network.gio.supported-protocols", "");


/******************************************************************************
 * SECTION 6: TLS/SSL HARDENING
 * Enforce modern cryptographic standards
 ******************************************************************************/

// Minimum TLS version (1.2 minimum, 1.3 preferred)
user_pref("security.tls.version.min", 3);
user_pref("security.tls.version.max", 4);
user_pref("security.tls.version.enable-deprecated", false);

// Disable 0-RTT (replay attack risk)
user_pref("security.tls.enable_0rtt_data", false);

// SSL3 disabled (obsolete)
user_pref("security.ssl3.rsa_des_ede3_sha", false);

// Require safe TLS renegotiation
user_pref("security.ssl.require_safe_negotiation", true);
user_pref("security.ssl.treat_unsafe_negotiation_as_broken", true);

// OCSP - verify certificate revocation
user_pref("security.OCSP.enabled", 1);
user_pref("security.OCSP.require", true);

// Certificate Transparency
user_pref("security.pki.crlite_mode", 2);

// Disable SHA-1 certificates
user_pref("security.pki.sha1_enforcement_level", 1);

// Don't trust third-party root CAs automatically
// user_pref("security.enterprise_roots.enabled", false);

// Certificate pinning enforcement
user_pref("security.cert_pinning.enforcement_level", 2);

// Enable CRLite for certificate revocation
user_pref("security.remote_settings.crlite_filters.enabled", true);


/******************************************************************************
 * SECTION 7: REFERRER POLICY HARDENING
 * Control what information is sent in Referer headers
 ******************************************************************************/

// Referrer header policy
// 0=full URI, 1=scheme+host+port+path, 2=scheme+host+port
user_pref("network.http.referer.trimmingPolicy", 2);

// Cross-origin referrer policy
// 0=always, 1=same-base-domain, 2=same-host
user_pref("network.http.referer.XOriginPolicy", 2);
user_pref("network.http.referer.XOriginTrimmingPolicy", 2);

// Send referrer header
// 0=never, 1=send for clicks, 2=send for images too
user_pref("network.http.sendRefererHeader", 2);

// Spoof referrer source
user_pref("network.http.referer.spoofSource", false);

// Default referrer policy for documents
user_pref("network.http.referer.defaultPolicy", 2);
user_pref("network.http.referer.defaultPolicy.pbmode", 2);

// Disable meta referrer
user_pref("network.http.referer.disallowCrossSiteRelaxingDefault", true);
user_pref("network.http.referer.disallowCrossSiteRelaxingDefault.top_navigation", true);


/******************************************************************************
 * SECTION 8: TELEMETRY & DATA COLLECTION ELIMINATION
 * Disable ALL phone-home functionality
 ******************************************************************************/

// [CRITICAL] Master telemetry switches
user_pref("toolkit.telemetry.enabled", false);
user_pref("toolkit.telemetry.unified", false);
user_pref("toolkit.telemetry.archive.enabled", false);
user_pref("toolkit.telemetry.server", "data:,");
user_pref("toolkit.telemetry.newProfilePing.enabled", false);
user_pref("toolkit.telemetry.updatePing.enabled", false);
user_pref("toolkit.telemetry.bhrPing.enabled", false);
user_pref("toolkit.telemetry.firstShutdownPing.enabled", false);
user_pref("toolkit.telemetry.shutdownPingSender.enabled", false);
user_pref("toolkit.telemetry.pioneer-new-studies-available", false);
user_pref("toolkit.telemetry.cachedClientID", "");
user_pref("toolkit.telemetry.coverage.opt-out", true);

// Health reports & data policy
user_pref("datareporting.healthreport.uploadEnabled", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);

// Crash reports
user_pref("breakpad.reportURL", "");
user_pref("browser.tabs.crashReporting.sendReport", false);
user_pref("browser.crashReports.unsubmittedCheck.autoSubmit2", false);

// Studies / Normandy / Shield experiments
user_pref("app.shield.optoutstudies.enabled", false);
user_pref("app.normandy.enabled", false);
user_pref("app.normandy.api_url", "");
user_pref("app.normandy.first_run", false);

// Ping centre
user_pref("browser.ping-centre.telemetry", false);

// Coverage endpoint
user_pref("toolkit.coverage.endpoint.base", "");
user_pref("toolkit.coverage.opt-out", true);

// Disable beacon API (analytics tracking)
user_pref("beacon.enabled", false);


/******************************************************************************
 * SECTION 9: SAFE BROWSING (SECURITY vs PRIVACY TRADE-OFF)
 * Google's malware/phishing database - sends hashes to remote servers
 * PARANOID: Disable all | BALANCED: Keep downloads check
 ******************************************************************************/

// [PARANOID MODE] Disable all Safe Browsing
user_pref("browser.safebrowsing.malware.enabled", false);
user_pref("browser.safebrowsing.phishing.enabled", false);
user_pref("browser.safebrowsing.blockedURIs.enabled", false);
user_pref("browser.safebrowsing.downloads.enabled", false);
user_pref("browser.safebrowsing.downloads.remote.enabled", false);
user_pref("browser.safebrowsing.downloads.remote.url", "");
user_pref("browser.safebrowsing.downloads.remote.block_potentially_unwanted", false);
user_pref("browser.safebrowsing.downloads.remote.block_uncommon", false);
user_pref("browser.safebrowsing.provider.google4.gethashURL", "");
user_pref("browser.safebrowsing.provider.google4.updateURL", "");
user_pref("browser.safebrowsing.provider.google.gethashURL", "");
user_pref("browser.safebrowsing.provider.google.updateURL", "");

// Disable password breach checks
user_pref("signon.management.page.breach-alerts.enabled", false);
user_pref("extensions.fxmonitor.enabled", false);


/******************************************************************************
 * SECTION 10: GEOLOCATION, PERMISSIONS & SENSORS
 * Lock down hardware access and location services
 ******************************************************************************/

// [CRITICAL] Disable geolocation entirely
user_pref("geo.enabled", false);
user_pref("geo.provider.network.url", "");
user_pref("geo.provider.ms-windows-location", false);
user_pref("geo.provider.use_corelocation", false);
user_pref("geo.provider.use_gpsd", false);
user_pref("geo.provider.use_geoclue", false);

// Disable geo IP lookups
user_pref("browser.search.geoip.url", "");
user_pref("browser.search.region", "US");

// Camera and microphone - always block
user_pref("permissions.default.camera", 2);
user_pref("permissions.default.microphone", 2);
user_pref("permissions.default.desktop-notification", 2);
user_pref("permissions.default.xr", 2);
user_pref("permissions.default.geo", 2);

// Web notifications (push spam)
user_pref("dom.webnotifications.enabled", false);
user_pref("dom.webnotifications.serviceworker.enabled", false);

// Disable speech recognition/synthesis
user_pref("media.webspeech.recognition.enable", false);
user_pref("media.webspeech.synth.enabled", false);

// Disable Web MIDI (hardware fingerprinting)
user_pref("dom.webmidi.enabled", false);

// Disable USB access
user_pref("dom.usb.enabled", false);

// Disable Bluetooth
user_pref("dom.bluetooth.enabled", false);

// Disable NFC
user_pref("dom.nfc.enabled", false);


/******************************************************************************
 * SECTION 11: WEBGL & GPU FINGERPRINTING
 * Graphics API is a major fingerprinting vector
 ******************************************************************************/

// [CRITICAL] Disable WebGL (may break some sites)
user_pref("webgl.disabled", true);

// If WebGL needed, harden instead:
// user_pref("webgl.disabled", false);
// user_pref("webgl.enable-debug-renderer-info", false);
// user_pref("webgl.enable-webgl2", false);
// user_pref("webgl.min_capability_mode", true);
// user_pref("webgl.disable-fail-if-major-performance-caveat", true);

// Disable GPU hardware acceleration (reduces fingerprint surface)
// WARNING: Impacts performance - uncomment only if paranoid
// user_pref("gfx.direct2d.disabled", true);
// user_pref("layers.acceleration.disabled", true);

// Disable GPU process sandbox for security
// user_pref("security.sandbox.gpu.level", 0);

// Canvas restrictions
user_pref("dom.webaudio.enabled", false);


/******************************************************************************
 * SECTION 12: MEDIA, DRM & CODEC FINGERPRINTING
 * Lock down media handling
 ******************************************************************************/

// Disable DRM (Netflix, Disney+, etc. will break)
user_pref("media.eme.enabled", false);
user_pref("media.gmp-widevinecdm.enabled", false);
user_pref("media.gmp-widevinecdm.visible", false);

// Disable GMP (Gecko Media Plugins)
user_pref("media.gmp-provider.enabled", false);
user_pref("media.gmp.storage.version.observed", 1);

// OpenH264 codec - disable
user_pref("media.gmp-gmpopenh264.enabled", false);
user_pref("media.gmp-gmpopenh264.autoupdate", false);

// Autoplay blocking (default 5 = block all)
user_pref("media.autoplay.default", 5);
user_pref("media.autoplay.blocking_policy", 2);
user_pref("media.autoplay.allow-extension-background-pages", false);
user_pref("media.autoplay.block-event.enabled", true);

// Disable Media Session API (exposes media state)
user_pref("dom.media.mediasession.enabled", false);

// Disable Picture-in-Picture toggle
user_pref("media.videocontrols.picture-in-picture.enabled", false);
user_pref("media.videocontrols.picture-in-picture.video-toggle.enabled", false);


/******************************************************************************
 * SECTION 13: SEARCH ENGINE & URL BAR PRIVACY
 * Prevent search suggestions from leaking queries
 ******************************************************************************/

// Disable live search suggestions
user_pref("browser.search.suggest.enabled", false);
user_pref("browser.urlbar.suggest.searches", false);
user_pref("browser.urlbar.suggest.quicksuggest.sponsored", false);
user_pref("browser.urlbar.suggest.quicksuggest.nonsponsored", false);

// Disable trending suggestions
user_pref("browser.urlbar.trending.featureGate", false);
user_pref("browser.urlbar.suggest.trending", false);

// Disable remote suggestions
user_pref("browser.urlbar.quicksuggest.enabled", false);
user_pref("browser.urlbar.quicksuggest.dataCollection.enabled", false);

// URL bar suggestions - local only
user_pref("browser.urlbar.suggest.history", false);
user_pref("browser.urlbar.suggest.bookmark", true);
user_pref("browser.urlbar.suggest.openpage", false);
user_pref("browser.urlbar.suggest.topsites", false);
user_pref("browser.urlbar.suggest.engines", false);
user_pref("browser.urlbar.suggest.calculator", true);
user_pref("browser.urlbar.maxRichResults", 5);

// Disable clipboard suggestions
user_pref("browser.urlbar.suggest.clipboard", false);

// Disable Urlbar addons suggestions
user_pref("browser.urlbar.addons.featureGate", false);

// Disable search engine geo-updates
user_pref("browser.search.update", false);
user_pref("browser.search.geoSpecificDefaults", false);
user_pref("browser.search.geoSpecificDefaults.url", "");


/******************************************************************************
 * SECTION 14: FORM AUTOFILL, PASSWORDS & CREDENTIALS
 * Disable browser credential storage - use external password manager
 ******************************************************************************/

// Disable form autofill
user_pref("browser.formfill.enable", false);
user_pref("extensions.formautofill.addresses.enabled", false);
user_pref("extensions.formautofill.creditCards.enabled", false);
user_pref("extensions.formautofill.heuristics.enabled", false);

// Disable built-in password manager
user_pref("signon.rememberSignons", false);
user_pref("signon.autofillForms", false);
user_pref("signon.formlessCapture.enabled", false);
user_pref("signon.privateBrowsingCapture.enabled", false);
user_pref("signon.generation.enabled", false);
user_pref("signon.management.page.breach-alerts.enabled", false);
user_pref("signon.firefoxRelay.feature", "disabled");

// Disable HTTP authentication credential storage
user_pref("network.auth.subresource-http-auth-allow", 1);

// Disable FIDO/WebAuthn (hardware keys - enable if you use them)
// user_pref("security.webauth.webauthn", false);
user_pref("security.webauth.u2f", true); // Keep for 2FA hardware keys

// Clipboard events - prevent websites from reading clipboard
user_pref("dom.event.clipboardevents.enabled", false);

// Disable async clipboard API
user_pref("dom.events.asyncClipboard.readText", false);
user_pref("dom.events.asyncClipboard.clipboardItem", false);


/******************************************************************************
 * SECTION 15: POCKET, NEW TAB, ACTIVITY STREAM
 * Disable Mozilla's recommendation services
 ******************************************************************************/

// Kill Pocket integration
user_pref("extensions.pocket.enabled", false);
user_pref("extensions.pocket.api", "");
user_pref("extensions.pocket.oAuthConsumerKey", "");
user_pref("extensions.pocket.site", "");

// New tab page - strip all Mozilla content
user_pref("browser.newtabpage.enabled", false);
user_pref("browser.newtabpage.activity-stream.enabled", false);
user_pref("browser.newtabpage.activity-stream.showSponsored", false);
user_pref("browser.newtabpage.activity-stream.showSponsoredTopSites", false);
user_pref("browser.newtabpage.activity-stream.section.highlights.includePocket", false);
user_pref("browser.newtabpage.activity-stream.section.highlights.includeDownloads", false);
user_pref("browser.newtabpage.activity-stream.section.highlights.includeVisited", false);
user_pref("browser.newtabpage.activity-stream.section.highlights.includeBookmarks", false);
user_pref("browser.newtabpage.activity-stream.feeds.section.highlights", false);
user_pref("browser.newtabpage.activity-stream.feeds.snippets", false);
user_pref("browser.newtabpage.activity-stream.feeds.section.topstories", false);
user_pref("browser.newtabpage.activity-stream.feeds.topsites", false);
user_pref("browser.newtabpage.activity-stream.feeds.system.topstories", false);
user_pref("browser.newtabpage.activity-stream.feeds.discoverystreamfeed", false);

// Activity stream telemetry
user_pref("browser.newtabpage.activity-stream.feeds.telemetry", false);
user_pref("browser.newtabpage.activity-stream.telemetry", false);
user_pref("browser.newtabpage.activity-stream.telemetry.structuredIngestion.endpoint", "");

// Disable top sites
user_pref("browser.topsites.contile.enabled", false);
user_pref("browser.topsites.contile.endpoint", "");

// Use blank home page
user_pref("browser.startup.homepage", "about:blank");
user_pref("browser.startup.page", 0);


/******************************************************************************
 * SECTION 16: DOWNLOADS & FILE HANDLING
 * Prevent automatic execution of downloaded files
 ******************************************************************************/

// Always ask where to save
user_pref("browser.download.useDownloadDir", false);

// Ask for each download type
user_pref("browser.download.always_ask_before_handling_new_types", true);

// Don't automatically open downloads
user_pref("browser.download.manager.addToRecentDocs", false);

// Disable external protocol handlers
user_pref("network.protocol-handler.external-default", false);
user_pref("network.protocol-handler.warn-external-default", true);

// Block specific protocol handlers
user_pref("network.protocol-handler.external.ms-windows-store", false);
user_pref("network.protocol-handler.external.mailto", false);
user_pref("network.protocol-handler.external.news", false);
user_pref("network.protocol-handler.external.snews", false);
user_pref("network.protocol-handler.external.nntp", false);

// Disable PDF.js (use external PDF reader for security)
// user_pref("pdfjs.disabled", true);
// If keeping PDF.js, disable scripting:
user_pref("pdfjs.enableScripting", false);


/******************************************************************************
 * SECTION 17: DOM SECURITY HARDENING
 * Restrict dangerous DOM features
 ******************************************************************************/

// Prevent popup window manipulation
user_pref("dom.disable_window_move_resize", true);
user_pref("dom.disable_window_flip", true);
user_pref("dom.disable_window_status_change", true);
user_pref("dom.disable_window_open_feature.close", true);
user_pref("dom.disable_window_open_feature.location", true);
user_pref("dom.disable_window_open_feature.menubar", true);
user_pref("dom.disable_window_open_feature.minimizable", true);
user_pref("dom.disable_window_open_feature.personalbar", true);
user_pref("dom.disable_window_open_feature.resizable", true);
user_pref("dom.disable_window_open_feature.status", true);
user_pref("dom.disable_window_open_feature.titlebar", true);
user_pref("dom.disable_window_open_feature.toolbar", true);

// Popup blocker
user_pref("dom.popup_maximum", 3);
user_pref("dom.popup_allowed_events", "click dblclick mousedown pointerdown");
user_pref("privacy.popups.disable_from_plugins", 2);

// Block onClick inline JavaScript links
// user_pref("browser.urlbar.filter.javascript", true);

// Disable SharedArrayBuffer (Spectre mitigation)
// Note: Modern Firefox already restricts this in cross-origin isolation
user_pref("javascript.options.shared_memory", false);

// Disable WebAssembly (can be used to fingerprint, execute exploits)
// WARNING: May break some legitimate web apps
user_pref("javascript.options.wasm", false);

// Disable Ion JIT (some exploits target JIT)
// WARNING: Significant performance impact
// user_pref("javascript.options.ion", false);
// user_pref("javascript.options.baselinejit", false);

// Reduce timing precision (fingerprinting mitigation)
user_pref("privacy.reduceTimerPrecision", true);
user_pref("privacy.resistFingerprinting.reduceTimerPrecision.microseconds", 1000);
user_pref("privacy.resistFingerprinting.reduceTimerPrecision.jitter", true);

// Disable Resource/User Timing API
user_pref("dom.enable_resource_timing", false);
user_pref("dom.enable_user_timing", false);
user_pref("dom.enable_performance", false);
user_pref("dom.enable_performance_observer", false);
user_pref("dom.enable_performance_navigation_timing", false);

// Disable Navigation/Performance API
user_pref("dom.performance.time_to_non_blank_paint.enabled", false);
user_pref("dom.performance.time_to_contentful_paint.enabled", false);
user_pref("dom.performance.time_to_first_interactive.enabled", false);

// PerformanceObserver - disable (used for timing attacks)
user_pref("dom.enable_performance", false);


/******************************************************************************
 * SECTION 18: EXTENSIONS & PLUGINS SECURITY
 * Lock down extension behavior
 ******************************************************************************/

// Require signed extensions (AMO signature)
user_pref("xpinstall.signatures.required", true);

// Block extension installs from non-AMO sources
user_pref("extensions.enabledScopes", 1);
user_pref("extensions.autoDisableScopes", 15);

// Disable add-on recommendations
user_pref("extensions.getAddons.showPane", false);
user_pref("extensions.htmlaboutaddons.recommendations.enabled", false);

// Disable Flash (it's dead anyway)
user_pref("plugin.state.flash", 0);

// Disable Java plugin
user_pref("plugin.state.java", 0);

// Block all plugins by default
user_pref("plugin.default.state", 0);

// Don't scan for plugins
user_pref("plugin.scan.plid.all", false);


/******************************************************************************
 * SECTION 19: UI SECURITY & ANTI-PHISHING
 * Prevent UI spoofing attacks
 ******************************************************************************/

// Display punycode for internationalized domains (IDN homograph attacks)
user_pref("network.IDN_show_punycode", true);

// Show full URL in address bar
user_pref("browser.urlbar.trimURLs", false);
user_pref("browser.urlbar.trimHttps", false);

// Highlight security-sensitive parts of URL
user_pref("browser.urlbar.formatting.enabled", true);

// Prevent fullscreen spoofing
user_pref("full-screen-api.warning.delay", 500);
user_pref("full-screen-api.warning.timeout", 3000);

// Block context menu override
user_pref("dom.event.contextmenu.enabled", false);

// Disable accessibility services (can be used for fingerprinting/attacks)
user_pref("accessibility.force_disabled", 1);
user_pref("accessibility.blockautorefresh", true);

// Prevent sites from overriding keyboard shortcuts
user_pref("permissions.default.shortcuts", 2);

// Warn on closing multiple tabs
user_pref("browser.tabs.warnOnClose", true);
user_pref("browser.tabs.warnOnCloseOtherTabs", true);

// Warn when opening about:config
user_pref("browser.aboutConfig.showWarning", false);


/******************************************************************************
 * SECTION 20: LINK DECORATION & TRACKING PREVENTION
 * Strip tracking parameters from URLs
 ******************************************************************************/

// Query stripping (removes fbclid, utm_*, etc.)
user_pref("privacy.query_stripping.enabled", true);
user_pref("privacy.query_stripping.enabled.pbmode", true);

// Custom strip list (add your own)
user_pref("privacy.query_stripping.strip_list", "fbclid fb_action_ids fb_action_types fb_source fb_ref mc_eid ml_subscriber ml_subscriber_hash msclkid oly_anon_id oly_enc_id rb_clickid s_cid vero_conv vero_id wickedid yclid __hssc __hstc __hsfp hsCtaTracking igshid utm_source utm_medium utm_term utm_content utm_campaign utm_id utm_name gclid gclsrc dclid zanpid _ga _gl ref twclid fbadid vgo_ee");

// Disable ping attribute (click tracking)
user_pref("browser.send_pings", false);
user_pref("browser.send_pings.require_same_host", true);

// Block Hyperlink Auditing
user_pref("browser.send_pings.max_per_link", 0);

// Strip referrer from links
user_pref("browser.urlbar.decodeURLsOnCopy", true);


/******************************************************************************
 * SECTION 21: FONT FINGERPRINTING PROTECTION
 * Limit font enumeration
 ******************************************************************************/

// Only use document fonts (blocks system font fingerprinting)
user_pref("browser.display.use_document_fonts", 0);

// Limit font visibility
user_pref("layout.css.font-visibility.private", 1);
user_pref("layout.css.font-visibility.standard", 1);
user_pref("layout.css.font-visibility.trackingprotection", 1);

// Block downloadable fonts (may break some sites)
// user_pref("gfx.downloadable_fonts.enabled", false);

// Disable WOFF2 fonts
// user_pref("gfx.downloadable_fonts.woff2.enabled", false);


/******************************************************************************
 * SECTION 22: FISSION (SITE ISOLATION)
 * Process isolation for security (similar to Chrome's site isolation)
 ******************************************************************************/

// Enable Fission (process per site)
user_pref("fission.autostart", true);

// WebExtension content script process isolation
user_pref("extensions.webextensions.remote", true);

// Separate process for file:// URIs
user_pref("browser.tabs.remote.separateFileUriProcess", true);

// GPU process sandbox
user_pref("security.sandbox.gpu.level", 1);

// Content process sandbox (Linux)
user_pref("security.sandbox.content.level", 4);

// Socket process isolation
user_pref("network.process.enabled", true);


/******************************************************************************
 * SECTION 23: CONTAINERS & IDENTITY SEPARATION
 * Settings for Multi-Account Containers extension
 ******************************************************************************/

// Enable containers
user_pref("privacy.userContext.enabled", true);
user_pref("privacy.userContext.ui.enabled", true);

// Containers for new tab page
user_pref("privacy.userContext.newTabContainerOnLeftClick.enabled", false);

// Long-press to choose container
user_pref("privacy.userContext.longPressBehavior", 2);


/******************************************************************************
 * SECTION 24: MISCELLANEOUS HARDENING
 * Additional security settings
 ******************************************************************************/

// Disable middle mouse paste (Linux security)
user_pref("middlemouse.paste", false);
user_pref("middlemouse.contentLoadURL", false);

// Disable UI animations (fingerprinting + privacy)
user_pref("ui.prefersReducedMotion", 1);

// Disable system colors (fingerprinting)
user_pref("browser.display.use_system_colors", false);

// Override color scheme to light (consistency)
user_pref("browser.theme.content-theme", 1);
user_pref("browser.theme.toolbar-theme", 1);

// Disable reader view (local processing is fine but disable if paranoid)
// user_pref("reader.parse-on-load.enabled", false);

// Disable screenshot tool
user_pref("extensions.screenshots.disabled", true);

// Disable Remote Debugging
user_pref("devtools.debugger.remote-enabled", false);
user_pref("devtools.chrome.enabled", false);
user_pref("devtools.debugger.force-local", true);

// Disable page thumbnails
user_pref("browser.pagethumbnails.capturing_disabled", true);

// Disable session restore
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.sessionstore.max_tabs_undo", 0);
user_pref("browser.sessionstore.max_windows_undo", 0);
user_pref("browser.sessionstore.privacy_level", 2);

// Disable background/preloaded tabs
user_pref("browser.tabs.loadInBackground", true);
user_pref("browser.newtab.preload", false);

// Disable window.name tracking
user_pref("privacy.window.name.update.enabled", true);

// Isolate window.opener
user_pref("dom.targetBlankNoOpener.enabled", true);

// Disable CSS :visited link history sniffing
user_pref("layout.css.visited_links_enabled", false);

// Disable payment request API
user_pref("dom.payments.defaults.saveAddress", false);
user_pref("dom.payments.defaults.saveCreditCard", false);
user_pref("dom.payments.request.enabled", false);

// Disable autoscroll
user_pref("general.autoScroll", false);

// Disable smooth scrolling (fingerprinting via scrolling behavior)
// user_pref("general.smoothScroll", false);


/******************************************************************************
 * SECTION 25: DEPRECATED/LEGACY PREFS (Keep for older Firefox versions)
 * These may not work on newer Firefox but kept for compatibility
 ******************************************************************************/

// Legacy prefs - may be deprecated
user_pref("network.http.referer.hideOnionSource", true);
user_pref("privacy.userContext.extension", "");


/******************************************************************************
 * SECTION 26: CUSTOM USER OVERRIDES
 * Add your site-specific or workflow-specific settings here
 ******************************************************************************/

/*
 * EXAMPLE: Re-enable specific features for compatibility
 *
 * // If you need WebGL for specific work:
 * user_pref("webgl.disabled", false);
 *
 * // If you need DRM for streaming:
 * user_pref("media.eme.enabled", true);
 * user_pref("media.gmp-widevinecdm.enabled", true);
 *
 * // If you need JavaScript for work:
 * user_pref("javascript.enabled", true);
 *
 * // If you use Tor as proxy:
 * user_pref("network.proxy.type", 1);
 * user_pref("network.proxy.socks", "127.0.0.1");
 * user_pref("network.proxy.socks_port", 9050);
 * user_pref("network.proxy.socks_remote_dns", true);
 */


/******************************************************************************
 * END OF CONFIGURATION
 * 
 * Post-deployment checklist:
 * [ ] Verify settings: about:config
 * [ ] Test for leaks: browserleaks.com, coveryourtracks.eff.org
 * [ ] Install recommended extensions
 * [ ] Test critical workflows
 * [ ] Create backup profile for incompatible sites
 * [ ] Set up container tabs for sensitive tasks
 * 
 * Remember: This config prioritizes privacy over convenience.
 * Some things WILL break. That's a feature, not a bug.
 ******************************************************************************/