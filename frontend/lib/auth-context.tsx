"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { auth } from "./firebase";
import { onAuthStateChanged, User, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut, updateProfile } from "firebase/auth";

interface AuthContextType {
  user: User | null;
  demoUser: { uid: string; email: string; name: string; role: string } | null;
  isAuthenticated: boolean;
  loading: boolean;
  loginDemo: (role?: string) => void;
  loginWithVerifiedOtp: (uid: string, token: string, email: string, name: string) => void;
  loginFirebase: (email: string, pass: string) => Promise<void>;
  registerFirebase: (email: string, pass: string, fullName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  demoUser: null,
  isAuthenticated: false,
  loading: true,
  loginDemo: () => {},
  loginWithVerifiedOtp: () => {},
  loginFirebase: async () => {},
  registerFirebase: async () => {},
  logout: () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [demoUser, setDemoUser] = useState<{ uid: string; email: string; name: string; role: string } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Check if demo token exists in localStorage
    const savedDemo = localStorage.getItem("healthtrack_demo_user");
    if (savedDemo) {
      try {
        setDemoUser(JSON.parse(savedDemo));
      } catch {
        localStorage.removeItem("healthtrack_demo_user");
      }
    }

    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      if (firebaseUser) {
        localStorage.setItem("healthtrack_last_uid", firebaseUser.uid);
        localStorage.removeItem("healthtrack_demo_user");
        localStorage.removeItem("healthtrack_demo_token");
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const loginDemo = (role: string = "patient_alpha") => {
    const demo = {
      uid: role,
      email: `${role}@healthtrack.ai`,
      name: role === "patient_alpha" ? "Self (Jane Doe)" : role === "mem_dad" ? "Dad (Robert Doe)" : "Patient",
      role: role,
    };
    localStorage.setItem("healthtrack_demo_token", `dev_${role}`);
    localStorage.setItem("healthtrack_demo_user", JSON.stringify(demo));
    localStorage.setItem("healthtrack_last_uid", role);
    setDemoUser(demo);
  };

  const loginWithVerifiedOtp = (uid: string, token: string, email: string, name: string) => {
    const verifiedUser = {
      uid: uid,
      email: email,
      name: name || "Patient",
      role: "PATIENT",
    };
    localStorage.setItem("healthtrack_demo_token", token);
    localStorage.setItem("healthtrack_demo_user", JSON.stringify(verifiedUser));
    localStorage.setItem("healthtrack_last_uid", uid);
    setDemoUser(verifiedUser);
  };

  const loginFirebase = async (email: string, pass: string) => {
    const cred = await signInWithEmailAndPassword(auth, email, pass);
    setUser(cred.user);
    setDemoUser(null);
    localStorage.setItem("healthtrack_last_uid", cred.user.uid);
    localStorage.removeItem("healthtrack_demo_user");
    localStorage.removeItem("healthtrack_demo_token");
  };

  const registerFirebase = async (email: string, pass: string, fullName?: string) => {
    const cred = await createUserWithEmailAndPassword(auth, email, pass);
    if (fullName && fullName.trim()) {
      try {
        await updateProfile(cred.user, { displayName: fullName.trim() });
      } catch (e) {
        console.warn("Could not set displayName:", e);
      }
    }
    setUser(cred.user);
    setDemoUser(null);
    localStorage.setItem("healthtrack_last_uid", cred.user.uid);
    localStorage.removeItem("healthtrack_demo_user");
    localStorage.removeItem("healthtrack_demo_token");
  };

  const logout = () => {
    signOut(auth);
    setUser(null);
    setDemoUser(null);
    localStorage.removeItem("healthtrack_demo_token");
    localStorage.removeItem("healthtrack_demo_user");
    localStorage.removeItem("healthtrack_last_uid");
  };

  const isAuthenticated = !!user || !!demoUser;

  return (
    <AuthContext.Provider
      value={{
        user,
        demoUser,
        isAuthenticated,
        loading,
        loginDemo,
        loginWithVerifiedOtp,
        loginFirebase,
        registerFirebase,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

