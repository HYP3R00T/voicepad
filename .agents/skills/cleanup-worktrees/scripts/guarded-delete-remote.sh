#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s <branch> <expected-object-id>\n' "$0" >&2
  exit 64
fi

branch=$1
expected_oid=$2
target_ref="refs/heads/$branch"

if ! git check-ref-format "$target_ref"; then
  printf 'invalid branch ref: %s\n' "$target_ref" >&2
  exit 64
fi
if [[ ! $expected_oid =~ ^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$ ]]; then
  printf 'invalid expected object ID\n' >&2
  exit 64
fi
expected_oid=${expected_oid,,}

if ! remote_output=$(git ls-remote --heads origin "$target_ref"); then
  printf 'failed to read origin ref: %s\n' "$target_ref" >&2
  exit 1
fi
remote_matches=()
if [[ -n $remote_output ]]; then
  mapfile -t remote_matches <<<"$remote_output"
fi
if [[ ${#remote_matches[@]} -ne 1 ]]; then
  printf 'expected exactly one origin ref for %s, found %d\n' "$target_ref" "${#remote_matches[@]}" >&2
  exit 1
fi
read -r observed_oid observed_ref <<<"${remote_matches[0]}"
if [[ $observed_ref != "$target_ref" || ${observed_oid,,} != "$expected_oid" ]]; then
  printf 'origin ref changed: expected %s at %s, observed %s at %s\n' \
    "$target_ref" "$expected_oid" "$observed_ref" "$observed_oid" >&2
  exit 1
fi

umask 077
hook_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "$hook_dir"
}
trap cleanup EXIT
chmod 700 "$hook_dir"

cat >"$hook_dir/pre-push" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

count=0
while read -r local_ref local_oid remote_ref remote_oid; do
  count=$((count + 1))
  if [[ $local_ref != "(delete)" || $local_oid =~ [^0] || $remote_ref != "$CLEANUP_TARGET_REF" || ${remote_oid,,} != "$CLEANUP_EXPECTED_OID" ]]; then
    printf 'refusing unexpected push: %s %s %s %s\n' "$local_ref" "$local_oid" "$remote_ref" "$remote_oid" >&2
    exit 1
  fi
done

if [[ $count -ne 1 ]]; then
  printf 'refusing push with %d ref updates\n' "$count" >&2
  exit 1
fi
HOOK
chmod 700 "$hook_dir/pre-push"

export CLEANUP_TARGET_REF=$target_ref
export CLEANUP_EXPECTED_OID=$expected_oid
git -c "core.hooksPath=$hook_dir" push origin ":$target_ref"

set +e
git ls-remote --exit-code --heads origin "$target_ref" >/dev/null 2>&1
verify_status=$?
set -e
case $verify_status in
  2) ;;
  0)
    printf 'origin ref still exists after deletion: %s\n' "$target_ref" >&2
    exit 1
    ;;
  *)
    printf 'failed to verify origin ref deletion: %s\n' "$target_ref" >&2
    exit 1
    ;;
esac
