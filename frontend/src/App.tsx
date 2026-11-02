import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ApiGate } from "./components/ApiGate";
import { Layout } from "./components/Layout";
import { Skeleton } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview"));
const Predict = lazy(() => import("./pages/Predict"));
const Customers = lazy(() => import("./pages/Customers"));
const Roi = lazy(() => import("./pages/Roi"));
const Batch = lazy(() => import("./pages/Batch"));
const ModelCard = lazy(() => import("./pages/ModelCard"));

function PageFallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-12 w-72" />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
      <Skeleton className="h-80" />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ApiGate>
        {(health) => (
          <Routes>
            <Route element={<Layout health={health} />}>
              {(
                [
                  ["/", Overview],
                  ["/predict", Predict],
                  ["/customers", Customers],
                  ["/roi", Roi],
                  ["/batch", Batch],
                  ["/model", ModelCard],
                ] as const
              ).map(([path, Page]) => (
                <Route
                  key={path}
                  path={path}
                  element={
                    <Suspense fallback={<PageFallback />}>
                      <Page />
                    </Suspense>
                  }
                />
              ))}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        )}
      </ApiGate>
    </BrowserRouter>
  );
}
