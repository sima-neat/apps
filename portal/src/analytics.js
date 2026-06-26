// Read-only analytics for the Examples (Apps Portal) section.
//
// The Developer Center stores one cookie-consent decision in localStorage on
// the shared developer.sima.ai origin (written by the docs/core consent
// banners). This module only *reads* that decision — it never renders a banner
// and never writes consent. GA4 is loaded only when analytics consent is
// already granted, so undecided/denied visitors are never tracked.
//
// The measurement ID is injected at build time from the per-env Vulcan config
// (analytics.ga_measurement_id) via Vite `define` (see vite.config.js), so
// examples reports into the same GA4 property as the rest of the Developer
// Center: staging G-T1D371DW1K, prod G-Y6YREB4EG5.

const CONSENT_KEY = "sima-developer-center-cookie-consent";
const CONSENT_VERSION = 1;

// Replaced at build time by Vite `define`. Falls back to "" when unset
// (local dev, or a config bundle without the key) so gtag never loads.
const MEASUREMENT_ID =
  typeof __PORTAL_GA_MEASUREMENT_ID__ !== "undefined" ? __PORTAL_GA_MEASUREMENT_ID__ : "";

const deniedConsent = {
  ad_storage: "denied",
  ad_user_data: "denied",
  ad_personalization: "denied",
  analytics_storage: "denied",
};

const grantedConsent = {
  ...deniedConsent,
  analytics_storage: "granted",
};

let gtagLoaded = false;
let lastTrackedLocation = "";

function getStoredConsent() {
  try {
    const value = window.localStorage.getItem(CONSENT_KEY);
    const parsed = value ? JSON.parse(value) : null;
    if (!parsed || parsed.version !== CONSENT_VERSION) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function analyticsGranted() {
  return getStoredConsent()?.analytics === true;
}

function currentPath() {
  return `${window.location.pathname}${window.location.search}`;
}

function ensureGtag() {
  window.dataLayer = window.dataLayer || [];
  if (!window.gtag) {
    window.gtag = function gtag() {
      window.dataLayer.push(arguments);
    };
  }
  window.gtag("consent", "default", deniedConsent);
}

export function trackPageView() {
  if (!window.gtag || !analyticsGranted()) {
    return;
  }
  const path = currentPath();
  if (path === lastTrackedLocation) {
    return;
  }
  lastTrackedLocation = path;
  window.gtag("event", "page_view", {
    page_title: document.title,
    page_location: window.location.href,
    page_path: path,
    page_section: "examples",
  });
}

export function loadGtag() {
  if (gtagLoaded || !MEASUREMENT_ID || !analyticsGranted()) {
    return;
  }
  gtagLoaded = true;

  ensureGtag();
  window.gtag("consent", "update", grantedConsent);

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(MEASUREMENT_ID)}`;
  document.head.appendChild(script);

  window.gtag("js", new Date());
  window.gtag("config", MEASUREMENT_ID, {
    anonymize_ip: true,
    send_page_view: false,
  });

  trackPageView();
}
