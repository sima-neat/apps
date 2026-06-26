import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, useLocation } from "react-router-dom";
import App from "./App";
import { loadGtag, trackPageView } from "./analytics";
import "./styles.css";

const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

// Read-only analytics: loads GA4 only if consent was already granted elsewhere
// in the Developer Center, then records a page_view on first render and on each
// route change. Renders nothing.
function RouteAnalytics() {
  const location = useLocation();
  useEffect(() => {
    loadGtag();
    trackPageView();
  }, [location.pathname, location.search]);
  return null;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      <RouteAnalytics />
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
