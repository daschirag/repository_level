import { motion } from "framer-motion";
import type { ReactNode } from "react";

export function Skeleton({
  width = "100%",
  height = 14,
  radius = 6,
  className = "",
}: {
  width?: number | string;
  height?: number | string;
  radius?: number;
  className?: string;
}) {
  return (
    <span
      className={`skeleton ${className}`}
      style={{ width, height, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}

export function ErrorBanner({
  title,
  message,
  onRetry,
  retrying = false,
}: {
  title: string;
  message: string;
  onRetry?: () => void;
  retrying?: boolean;
}) {
  return (
    <motion.div
      className="error-banner"
      role="alert"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
    >
      <svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true">
        <path
          fill="currentColor"
          d="M10 1.8 19 17.5H1L10 1.8Zm0 5.2a.9.9 0 0 0-.9.9v4.3a.9.9 0 1 0 1.8 0V7.9A.9.9 0 0 0 10 7Zm0 8.6a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
        />
      </svg>
      <div className="error-banner__body">
        <strong>{title}</strong>
        <span>{message}</span>
      </div>
      {onRetry && (
        <button type="button" className="btn btn--ghost" onClick={onRetry} disabled={retrying}>
          {retrying ? "Retrying…" : "Retry"}
        </button>
      )}
    </motion.div>
  );
}

export function EmptyState({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="empty-state">
      <svg viewBox="0 0 64 64" width="56" height="56" aria-hidden="true">
        <circle cx="16" cy="20" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
        <circle cx="46" cy="16" r="5" fill="none" stroke="currentColor" strokeWidth="2" />
        <circle cx="34" cy="46" r="8" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M21 23l9 17M41 19l-5 19M22 19l19-3" stroke="currentColor" strokeWidth="2" strokeDasharray="3 4" />
      </svg>
      <h3>{title}</h3>
      <div className="empty-state__body">{children}</div>
    </div>
  );
}
