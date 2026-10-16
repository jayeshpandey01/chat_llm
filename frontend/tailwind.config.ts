import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        parchment: "#f3f0e8",
        darkCanvas: "#121212",
        darkCard: "#1a1a1a",
        darkCardHover: "#242424",
        accentBlue: {
          light: "#2f5df7",
          dark: "#5b82ff",
        },
        accentGreen: {
          light: "#00a86b",
          dark: "#26c485",
        },
        accentYellow: {
          light: "#fcd116",
          dark: "#ffe15d",
        },
        accentRed: {
          light: "#e63946",
          dark: "#ff5a5f",
        },
      },
      fontFamily: {
        logo: ["Syne", "sans-serif"],
        heading: ["Space Grotesk", "sans-serif"],
        body: ["Inter", "sans-serif"],
      },
      boxShadow: {
        brutal: "3px 3px 0px #000000",
        "brutal-lg": "5px 5px 0px #000000",
        "brutal-xl": "8px 8px 0px #000000",
        "brutal-dark": "3px 3px 0px #f3f0e8",
        "brutal-dark-lg": "5px 5px 0px #f3f0e8",
      },
      borderWidth: {
        "2": "2px",
        "3": "3px",
      },
    },
  },
  plugins: [],
};

export default config;

