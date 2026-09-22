import { useState } from "react";
import type { FollowUpAnswer, FollowUpQuestion } from "../types";

// Quick replies per question; the worker can also type a free answer.
const CHIPS: Record<string, string[]> = {
  rdt_result: ["Positive", "Negative", "Not done"],
  fever_days: ["1 day", "2 days", "3 days", "4 days", "5 days", "7+ days"],
  treatment_response: [
    "Took antimalarial, no better",
    "Took antibiotic, no better",
    "Took drugs, got better",
    "No treatment yet",
  ],
  age_years: ["Under 1 year", "1–4 years", "5–14 years", "Adult"],
};

export function Questions({
  questions,
  onSubmit,
  onSkip,
  disabled,
}: {
  questions: FollowUpQuestion[];
  onSubmit: (answers: FollowUpAnswer[]) => void;
  onSkip: () => void;
  disabled: boolean;
}) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const complete = questions.every((q) => answers[q.id]?.trim());
  const set = (id: string, value: string) => setAnswers((a) => ({ ...a, [id]: value }));

  return (
    <div className="space-y-4 rounded-xl border border-teal-200 bg-teal-50 p-3">
      {questions.map((q) => (
        <div key={q.id}>
          <p className="mb-2 font-semibold">{q.text}</p>
          <div className="flex flex-wrap gap-2">
            {(CHIPS[q.id] ?? []).map((chip) => (
              <button
                key={chip}
                type="button"
                disabled={disabled}
                onClick={() => set(q.id, chip)}
                className={`min-h-11 rounded-full border px-3 text-sm font-medium ${
                  answers[q.id] === chip ? "border-teal-800 bg-teal-800 text-white" : "border-teal-300 bg-white text-teal-900"
                }`}
              >
                {chip}
              </button>
            ))}
          </div>
          <input
            value={CHIPS[q.id]?.includes(answers[q.id]) ? "" : (answers[q.id] ?? "")}
            onChange={(e) => set(q.id, e.target.value)}
            placeholder="Or type an answer"
            disabled={disabled}
            className="mt-2 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 text-[15px]"
          />
        </div>
      ))}
      <div className="flex gap-2">
        <button
          type="button"
          disabled={!complete || disabled}
          onClick={() => onSubmit(questions.map((q) => ({ id: q.id, answer: answers[q.id].trim() })))}
          className="min-h-11 flex-1 rounded-lg bg-teal-800 px-3 font-semibold text-white disabled:opacity-40"
        >
          Continue
        </button>
        <button type="button" disabled={disabled} onClick={onSkip}
                className="min-h-11 rounded-lg border border-slate-300 bg-white px-3 font-semibold">
          Skip
        </button>
      </div>
    </div>
  );
}
