import Link from "next/link";

export default function NotFound() {
  return (
    <div className="p-12 text-center min-h-[400px] flex flex-col items-center justify-center">
      <div className="font-['Syne'] text-5xl font-extrabold uppercase mb-2">404</div>
      <h2 className="font-['Space_Grotesk'] text-xl font-bold uppercase mb-4">Page Not Found</h2>
      <p className="text-xs opacity-75 max-w-sm mb-6 font-['Inter']">
        The clinical route or consultation page you requested does not exist or has been archived.
      </p>
      <Link href="/" className="neo-btn neo-btn-primary">
        ← Return to Symptom Checker
      </Link>
    </div>
  );
}

