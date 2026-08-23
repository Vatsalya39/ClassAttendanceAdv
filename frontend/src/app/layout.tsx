import type { Metadata } from "next";
import localFont from "next/font/local";
import Link from "next/link";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Attendance — Face Recognition",
  description:
    "Multi-campus face-recognition attendance system. Matching is always scoped to a section roster.",
};

function NavLinks() {
  return (
    <nav className="navLinks">
      <Link className="navLink" href="/register">
        Register
      </Link>
      <Link className="navLink" href="/student/login">
        My Attendance
      </Link>
      <Link className="navLink" href="/admin/dashboard">
        Admin
      </Link>
    </nav>
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        <header className="nav">
          <div className="navInner">
            <Link className="brand" href="/">
              <span className="brandBadge">FR</span>
              Attendance
            </Link>
            <NavLinks />
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
