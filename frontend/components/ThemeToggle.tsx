"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const current = document.documentElement.getAttribute("data-theme") as "light" | "dark";
    if (current) setTheme(current);
  }, []);

  const toggle = () => {
    const html = document.documentElement;
    const current = html.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    setTheme(next);
  };

  return (
    <button
      id="themeToggleBtn"
      type="button"
      className="neo-btn !p-2"
      title={theme === "dark" ? "Switch to Light Theme" : "Switch to Dark Theme"}
      onClick={toggle}
      aria-label="Toggle Theme"
    >
      {theme === "dark" ? (
        <Sun className="w-4 h-4 stroke-[2.2]" />
      ) : (
        <Moon className="w-4 h-4 stroke-[2.2]" />
      )}
    </button>
  );
}
