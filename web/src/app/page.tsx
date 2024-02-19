import { getReadiness } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  const readiness = await getReadiness();

  return (
    <main className="mx-auto max-w-2xl px-4 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">ragGround</h1>
      <p className="mt-2 text-neutral-600 dark:text-neutral-400">
        Grounded answers over your documents, with citations and measured quality.
      </p>

      <section className="mt-10 rounded-lg border border-neutral-200 p-5 dark:border-neutral-800">
        <h2 className="text-sm font-medium uppercase tracking-wide text-neutral-500">
          System status
        </h2>
        {readiness.status === "unreachable" ? (
          <p className="mt-3 text-red-600">API unreachable: {readiness.error}</p>
        ) : (
          <ul className="mt-3 space-y-2">
            {Object.entries(readiness.checks).map(([name, check]) => (
              <li key={name} className="flex items-center justify-between gap-4 text-sm">
                <span className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className={`inline-block h-2 w-2 rounded-full ${check.ok ? "bg-green-500" : "bg-red-500"}`}
                  />
                  {name}
                </span>
                <span className="font-mono text-neutral-500">
                  {check.ok ? `${check.latency_ms} ms` : check.error}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
