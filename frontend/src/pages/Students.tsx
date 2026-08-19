/**
 * Student registry: search, inspect and delete enrolments.
 *
 * Deletion is the consent-withdrawal path, so it is behind a confirmation that
 * states plainly what is erased and that it cannot be undone.
 */

import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Student } from "../api/types";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Modal } from "../components/ui/Modal";
import {
  IconRefresh,
  IconSearch,
  IconTrash,
  IconUserPlus,
  IconUsers,
} from "../components/ui/icons";
import { EmptyState, ErrorState, Skeleton } from "../components/ui/States";
import { useToast } from "../components/ui/toast-context";
import { useAuth } from "../hooks/auth-context";
import { useMutation, useQuery } from "../hooks/useApi";
import { formatDateTime } from "../utils/format";
import "./Students.css";

export function Students() {
  const toast = useToast();
  // Deleting a student is admin-only on the API. Hiding the button for everyone
  // else keeps the UI honest — a control that always fails is worse than none.
  const { isAdmin } = useAuth();
  const [search, setSearch] = useState("");
  const [pendingDelete, setPendingDelete] = useState<Student | null>(null);

  const students = useQuery(() => api.listStudents(), []);
  const remove = useMutation(api.deleteStudent);

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    const all = students.data?.students ?? [];
    if (!term) return all;
    return all.filter(
      (student) =>
        student.name.toLowerCase().includes(term) ||
        student.student_id.toLowerCase().includes(term),
    );
  }, [students.data, search]);

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await remove.run(pendingDelete.student_id);
      toast.success(
        `${pendingDelete.name} deleted`,
        "Their face embeddings have been erased.",
      );
      setPendingDelete(null);
      students.refetch();
    } catch {
      toast.error("Could not delete the student", remove.error?.message);
    }
  };

  return (
    <div className="page">
      <PageHeader
        title="Students"
        description="Everyone enrolled in the face recognition system."
        actions={
          <>
            <Button icon={<IconRefresh size={15} />} onClick={students.refetch}>
              Refresh
            </Button>
            <Link to="/register" className="unstyled-link">
              <Button variant="primary" icon={<IconUserPlus size={15} />}>
                Register student
              </Button>
            </Link>
          </>
        }
      />

      {students.error ? (
        <div className="card">
          <ErrorState error={students.error} onRetry={students.refetch} />
        </div>
      ) : (
        <div className="card">
          <div className="table__toolbar">
            <div className="search">
              <IconSearch size={15} />
              <input
                className="search__input"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search by name or ID…"
                aria-label="Search students"
              />
            </div>
            <span className="table__count">
              {students.loading
                ? "Loading…"
                : `${filtered.length} student${filtered.length === 1 ? "" : "s"}`}
            </span>
          </div>

          {students.loading ? (
            <div className="table__skeleton">
              {[0, 1, 2, 3].map((key) => (
                <Skeleton key={key} height={52} radius="var(--radius)" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <EmptyState
              icon={<IconUsers size={22} />}
              title={search ? "No matching students" : "No students registered yet"}
              description={
                search
                  ? "Try a different name or ID."
                  : "Register a student's face so the system can recognise them."
              }
              action={
                !search && (
                  <Link to="/register" className="unstyled-link">
                    <Button variant="primary" icon={<IconUserPlus size={15} />}>
                      Register the first student
                    </Button>
                  </Link>
                )
              }
            />
          ) : (
            <div className="table__wrap">
              <table className="table">
                <caption className="sr-only">Registered students</caption>
                <thead>
                  <tr>
                    <th scope="col">Student</th>
                    <th scope="col">ID</th>
                    <th scope="col">Department</th>
                    <th scope="col">Face data</th>
                    <th scope="col">Registered</th>
                    {isAdmin && (
                      <th scope="col">
                        <span className="sr-only">Actions</span>
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((student) => (
                    <tr key={student.student_id}>
                      <td>
                        <div className="cell-user">
                          <span className="cell-user__avatar" aria-hidden="true">
                            {student.name.charAt(0).toUpperCase()}
                          </span>
                          <span className="cell-user__name">{student.name}</span>
                        </div>
                      </td>
                      <td className="tabular secondary">{student.student_id}</td>
                      <td className="secondary">
                        {(student.metadata.department as string) ?? "—"}
                      </td>
                      <td>
                        <Badge tone={student.embedding_count >= 3 ? "good" : "warning"}>
                          {student.embedding_count} embedding
                          {student.embedding_count === 1 ? "" : "s"}
                        </Badge>
                      </td>
                      <td className="secondary">{formatDateTime(student.created_at)}</td>
                      {isAdmin && (
                        <td className="table__actions-cell">
                          <Button
                            variant="ghost"
                            size="sm"
                            icon={<IconTrash size={14} />}
                            onClick={() => setPendingDelete(student)}
                            aria-label={`Delete ${student.name}`}
                          />
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <Modal
        open={pendingDelete !== null}
        title={`Delete ${pendingDelete?.name ?? ""}?`}
        description={`This permanently erases their face embeddings from the system. They will no longer be recognised and would need to register again. Their past attendance records are kept.`}
        confirmLabel="Delete permanently"
        danger
        loading={remove.loading}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
