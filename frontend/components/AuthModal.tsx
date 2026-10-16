"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { useRouter } from "next/navigation";
import { X } from "lucide-react";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultTab?: "login" | "register";
}

export default function AuthModal({ isOpen, onClose, defaultTab = "login" }: AuthModalProps) {
  const [tab, setTab] = useState<"login" | "register">(defaultTab);
  const [registerMode, setRegisterMode] = useState<"otp" | "password">("otp");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [otpCode, setOtpCode] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [otpInfo, setOtpInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const { loginFirebase, registerFirebase, loginDemo, loginWithVerifiedOtp } = useAuth();
  const router = useRouter();

  if (!isOpen) return null;

  const handleSendOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      setError("Please enter your email address.");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const res = await fetch("/api/auth/otp/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), name: fullName.trim() || "Patient" }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to dispatch verification code.");
      }
      setOtpSent(true);
      if (data.debug_code) {
        setOtpInfo(`Verification code dispatched! (Local Sandbox Code: ${data.debug_code})`);
      } else {
        setOtpInfo(`Verification code sent to ${email}. Valid for 10 minutes.`);
      }
    } catch (err: any) {
      setError(err.message || "Failed to send verification code. Please check your network.");
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!otpCode || otpCode.trim().length !== 6) {
      setError("Please enter the 6-digit verification code.");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const res = await fetch("/api/auth/otp/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: email.trim(),
          otp_code: otpCode.trim(),
          name: fullName.trim() || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Invalid or expired verification code.");
      }
      loginWithVerifiedOtp(data.uid, data.token, email.trim(), fullName.trim() || "Patient");
      onClose();
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.message || "OTP verification failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (tab === "login") {
        await loginFirebase(email, password);
      } else {
        await registerFirebase(email, password, fullName);
      }
      onClose();
      router.push("/dashboard");
    } catch (err: any) {
      setError(err.message || "Authentication failed. You can also use 1-Click Demo Mode below.");
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = (role: string) => {
    loginDemo(role);
    onClose();
    router.push("/dashboard");
  };

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="border-3 border-[var(--border-color)] bg-[var(--bg-color)] text-[var(--text-color)] max-w-md w-full p-6 shadow-[8px_8px_0px_var(--shadow-color)]">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-3 border-b-2 border-[var(--border-color)] mb-4">
          <div className="flex items-center gap-2">
            <span className="text-[var(--accent-blue)] font-bold text-lg">▲</span>
            <h3 className="font-['Space_Grotesk'] text-xl font-extrabold uppercase">
              {tab === "login" ? "Patient Sign In" : "Register Account"}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="hover:opacity-70 p-1"
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Toggle */}
        <div className="flex border-2 border-[var(--border-color)] bg-white text-black mb-4">
          <button
            type="button"
            onClick={() => { setTab("login"); setError(null); }}
            className={`flex-1 py-2 font-['Space_Grotesk'] text-xs font-bold uppercase transition-all ${
              tab === "login" ? "bg-[var(--accent-color)] text-[var(--accent-text)]" : "hover:bg-neutral-100"
            }`}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => { setTab("register"); setError(null); }}
            className={`flex-1 py-2 font-['Space_Grotesk'] text-xs font-bold uppercase transition-all border-l-2 border-[var(--border-color)] ${
              tab === "register" ? "bg-[var(--accent-color)] text-[var(--accent-text)]" : "hover:bg-neutral-100"
            }`}
          >
            Register
          </button>
        </div>

        {error && (
          <div className="p-3 bg-red-100 text-red-900 border-2 border-red-500 text-xs font-semibold mb-4 leading-snug">
            {error}
          </div>
        )}

        {otpInfo && tab === "register" && (
          <div className="p-3 bg-blue-50 text-blue-900 border-2 border-blue-500 text-xs font-semibold mb-4 leading-snug">
            {otpInfo}
          </div>
        )}

        {/* Register Mode Selector (OTP vs Password) */}
        {tab === "register" && (
          <div className="flex gap-2 mb-3 text-[11px] font-bold">
            <button
              type="button"
              onClick={() => { setRegisterMode("otp"); setError(null); }}
              className={`flex-1 py-1 px-2 border border-black ${registerMode === "otp" ? "bg-black text-white" : "bg-neutral-100 text-black hover:bg-neutral-200"}`}
            >
              ✉️ Email OTP (Recommended)
            </button>
            <button
              type="button"
              onClick={() => { setRegisterMode("password"); setError(null); }}
              className={`flex-1 py-1 px-2 border border-black ${registerMode === "password" ? "bg-black text-white" : "bg-neutral-100 text-black hover:bg-neutral-200"}`}
            >
              🔑 Password
            </button>
          </div>
        )}

        {/* OTP Registration Flow (Diagram 3) */}
        {tab === "register" && registerMode === "otp" ? (
          !otpSent ? (
            <form onSubmit={handleSendOtp} className="space-y-3 text-xs">
              <div>
                <label className="font-['Space_Grotesk'] font-bold uppercase block mb-1">Full Legal Name:</label>
                <input
                  type="text"
                  required
                  placeholder="Jane Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full p-2.5 border-2 border-[var(--border-color)] bg-white text-black font-['Inter'] outline-none"
                />
              </div>

              <div>
                <label className="font-['Space_Grotesk'] font-bold uppercase block mb-1">Email Address:</label>
                <input
                  type="email"
                  required
                  placeholder="patient@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full p-2.5 border-2 border-[var(--border-color)] bg-white text-black font-['Inter'] outline-none"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="neo-btn neo-btn-primary w-full justify-center !py-2.5 mt-2"
              >
                {loading ? "Sending Code..." : "Send Verification Code (Email OTP) →"}
              </button>
            </form>
          ) : (
            <form onSubmit={handleVerifyOtp} className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="font-['Space_Grotesk'] font-bold uppercase">Enter 6-Digit Code:</label>
                  <button
                    type="button"
                    onClick={() => { setOtpSent(false); setOtpInfo(null); setError(null); }}
                    className="text-[10px] text-blue-600 hover:underline font-bold"
                  >
                    Change Email
                  </button>
                </div>
                <input
                  type="text"
                  required
                  maxLength={6}
                  placeholder="123456"
                  value={otpCode}
                  onChange={(e) => setOtpCode(e.target.value)}
                  className="w-full p-3 border-2 border-black bg-white text-black text-center font-mono text-xl tracking-widest font-bold outline-none"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="neo-btn neo-btn-primary w-full justify-center !py-2.5 mt-2"
              >
                {loading ? "Verifying..." : "Verify & Access Telehealth →"}
              </button>
            </form>
          )
        ) : (
          /* Password Form (Login & Password Registration) */
          <form onSubmit={handleSubmit} className="space-y-3 text-xs">
            {tab === "register" && (
              <div>
                <label className="font-['Space_Grotesk'] font-bold uppercase block mb-1">Full Legal Name:</label>
                <input
                  type="text"
                  required
                  placeholder="Jane Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full p-2.5 border-2 border-[var(--border-color)] bg-white text-black font-['Inter'] outline-none"
                />
              </div>
            )}

            <div>
              <label className="font-['Space_Grotesk'] font-bold uppercase block mb-1">Email Address:</label>
              <input
                type="email"
                required
                placeholder="patient@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-2.5 border-2 border-[var(--border-color)] bg-white text-black font-['Inter'] outline-none"
              />
            </div>

            <div>
              <label className="font-['Space_Grotesk'] font-bold uppercase block mb-1">Password:</label>
              <input
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full p-2.5 border-2 border-[var(--border-color)] bg-white text-black font-mono outline-none"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="neo-btn neo-btn-primary w-full justify-center !py-2.5 mt-2"
            >
              {loading ? "Processing..." : tab === "login" ? "Sign In →" : "Create Account →"}
            </button>
          </form>
        )}

        {/* 1-Click Fast Evaluation Mode */}
        <div className="mt-5 pt-4 border-t-2 border-[var(--border-color)]">
          <div className="font-['Space_Grotesk'] text-[11px] font-bold uppercase mb-2 opacity-80 flex items-center justify-between">
            <span>Instant Evaluation Mode:</span>
            <span className="neo-badge neo-badge-green text-[9px]">1-Click Demo</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleDemoLogin("patient_alpha")}
              className="neo-btn !py-1.5 !px-2 !text-[11px] justify-center bg-white text-black hover:bg-neutral-100"
            >
              Demo: Self (Primary)
            </button>
            <button
              type="button"
              onClick={() => handleDemoLogin("mem_dad")}
              className="neo-btn !py-1.5 !px-2 !text-[11px] justify-center bg-white text-black hover:bg-neutral-100"
            >
              Demo: Dad (CAD History)
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}

