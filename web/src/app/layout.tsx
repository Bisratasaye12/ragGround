import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ragGround",
  description: "Grounded answers over your documents, with citations and measured quality.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
