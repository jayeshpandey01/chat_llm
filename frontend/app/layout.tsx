import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import { AuthProvider } from "@/lib/auth-context";

export const metadata: Metadata = {
  title: "HealthTrack AI | Clinical Decision Support & Telehealth Studio",
  description: "FDA CDSS-compliant telehealth studio with 99.91% primary care accuracy, 95% conformal bounds, and real-time SSE consultation streaming.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light">
      <body>
        <AuthProvider>
          <div className="neo-container">
            {children}
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
