#!/usr/bin/env bash
# A1 capacity auto-launcher — keeps trying to create the free Ampere A1 VM until
# capacity frees, then stops. This is how you land an A1 instance without
# clicking "Create" by hand for two days.
#
# PREREQUISITES (one-time, on your laptop or the micro box):
#   1. Install OCI CLI:  https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm
#        bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
#   2. Configure it:     oci setup config      (creates ~/.oci/config + an API key;
#        upload the public key under Console → your user → API Keys)
#   3. Gather the OCIDs below from the Console (Compute → Create Instance shows
#      them; or `oci iam compartment list`, `oci compute image list`, etc.)
#
# Fill these in, then:  chmod +x deploy/oci-launch-retry.sh && ./deploy/oci-launch-retry.sh
set -uo pipefail

# ── Fill in from the OCI Console (Mumbai region) ──────
COMPARTMENT_ID="ocid1.tenancy.oc1..xxxx"      # your tenancy/compartment OCID
SUBNET_ID="ocid1.subnet.oc1.ap-mumbai-1.xxxx" # the PUBLIC subnet from your VCN
IMAGE_ID="ocid1.image.oc1.ap-mumbai-1.xxxx"   # Ubuntu 22.04 (aarch64) image OCID
AVAILABILITY_DOMAIN="xxxx:AP-MUMBAI-1-AD-1"   # `oci iam availability-domain list`
SSH_PUBKEY_PATH="$HOME/.ssh/id_rsa.pub"       # your SSH public key
DISPLAY_NAME="trading-ai"

# ── A1 free shape config (2 OCPU / 12 GB) ─────────────
SHAPE="VM.Standard.A1.Flex"
OCPUS=2
MEM_GB=12

# ── Retry behaviour ───────────────────────────────────
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"          # wait between attempts
MAX_TRIES="${MAX_TRIES:-0}"                    # 0 = forever

command -v oci >/dev/null 2>&1 || { echo "OCI CLI not installed — see header."; exit 1; }
[ -f "$SSH_PUBKEY_PATH" ] || { echo "SSH public key not found: $SSH_PUBKEY_PATH"; exit 1; }

echo "Launching $SHAPE ($OCPUS OCPU / ${MEM_GB}G) in $AVAILABILITY_DOMAIN — retrying every ${SLEEP_SECONDS}s"
try=0
while :; do
  try=$((try+1))
  echo "[$(date '+%H:%M:%S')] attempt #$try ..."
  OUT=$(oci compute instance launch \
    --compartment-id "$COMPARTMENT_ID" \
    --availability-domain "$AVAILABILITY_DOMAIN" \
    --subnet-id "$SUBNET_ID" \
    --image-id "$IMAGE_ID" \
    --shape "$SHAPE" \
    --shape-config "{\"ocpus\": $OCPUS, \"memoryInGBs\": $MEM_GB}" \
    --assign-public-ip true \
    --display-name "$DISPLAY_NAME" \
    --ssh-authorized-keys-file "$SSH_PUBKEY_PATH" \
    --wait-for-state RUNNING 2>&1) && {
      echo "✅ SUCCESS — instance is RUNNING."
      echo "$OUT" | grep -i '"id"' | head -1
      echo "Get its public IP:  oci compute instance list-vnics --instance-id <id> --query 'data[0].\"public-ip\"'"
      exit 0
    }

  if echo "$OUT" | grep -qi "Out of host capacity\|Out of capacity\|InternalError"; then
    echo "   no capacity yet — sleeping ${SLEEP_SECONDS}s"
  else
    echo "   non-capacity error (check config):"; echo "$OUT" | tail -3
  fi
  [ "$MAX_TRIES" -gt 0 ] && [ "$try" -ge "$MAX_TRIES" ] && { echo "Hit MAX_TRIES=$MAX_TRIES, stopping."; exit 2; }
  sleep "$SLEEP_SECONDS"
done
