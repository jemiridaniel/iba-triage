import type { Meta, TriageRequest } from "./types";

/** POST /triage/stream and call onEvent for each server-sent event. */
export async function streamTriage(
  req: TriageRequest,
  onEvent: (event: string, data: any) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/triage/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(res.status === 422 ? "Please describe the patient in a few words." : `Server error (${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split: number;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

export async function getMeta(): Promise<Meta> {
  const res = await fetch("/meta");
  if (!res.ok) throw new Error(`Server error (${res.status})`);
  return res.json();
}
