/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for the Docker production stage (node server.js).
  // Azure Static Web Apps can consume the standalone output as well.
  output: "standalone",

  // Expose the API base URL to client-side code.
  // Set NEXT_PUBLIC_API_URL in the Container Apps / Static Web Apps env.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },

  // Allow images from the local API server (for face crop previews in the
  // review queue). Phase 4 adds the Azure Blob hostname here.
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/static/**",
      },
    ],
  },
};

export default nextConfig;
