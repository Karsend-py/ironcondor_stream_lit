
// public/app.js
// Wire up Authentication, Firestore (read), and Storage (upload)

import { auth, db, storage } from "./Firebase.js";

// ===== Auth imports =====
import {
  onAuthStateChanged,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// ===== Firestore imports =====
import {
  collection,
  query,
  orderBy,
  limit,
  getDocs
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

// ===== Storage imports =====
import {
  ref,
  uploadBytes
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-storage.js";

// ===== UI elements =====
const emailEl = document.getElementById("email");
const passEl  = document.getElementById("password");
const signUpBtn = document.getElementById("signUpBtn");
const signInBtn = document.getElementById("signInBtn");
const signOutBtn = document.getElementById("signOutBtn");
const barsList = document.getElementById("bars");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("fileInput");
const uploadBtn  = document.getElementById("uploadBtn");
const uploadStatusEl = document.getElementById("uploadStatus");

// ===== Authentication handlers =====
signUpBtn.addEventListener("click", async () => {
  try {
    const cred = await createUserWithEmailAndPassword(auth, emailEl.value, passEl.value);
    statusEl.textContent = `Signed up: ${cred.user.email}`;
    statusEl.className = "ok";
  } catch (err) {
    statusEl.textContent = `Sign up error: ${err.message}`;
    statusEl.className = "err";
  }
});

signInBtn.addEventListener("click", async () => {
  try {
    const cred = await signInWithEmailAndPassword(auth, emailEl.value, passEl.value);
    statusEl.textContent = `Signed in: ${cred.user.email}`;
    statusEl.className = "ok";
    await loadBars();
  } catch (err) {
    statusEl.textContent = `Sign in error: ${err.message}`;
    statusEl.className = "err";
  }
});

signOutBtn.addEventListener("click", async () => {
  await signOut(auth);
  statusEl.textContent = "Signed out";
  statusEl.className = "muted";
  barsList.innerHTML = "";
});

// Auth state listener (auto-load data when signed in)
onAuthStateChanged(auth, async (user) => {
  if (user) {
    statusEl.textContent = `Authenticated as ${user.email}`;
    statusEl.className = "ok";
    await loadBars();
  } else {
    statusEl.textContent = "Not signed in";
    statusEl.className = "muted";
    barsList.innerHTML = "<li class='muted'>Sign in to load data.</li>";
  }
});

// ===== Firestore: load first 10 AMZN bars =====
async function loadBars() {
  try {
    const barsCol = collection(db, "AMZN_daily-bars_2015-2025");
    const q = query(barsCol, orderBy("timestamp"), limit(10));
    const snap = await getDocs(q);

    barsList.innerHTML = "";
    if (snap.empty) {
      barsList.innerHTML = "<li>No documents found.</li>";
      return;
    }
    snap.forEach(doc => {
      const d = doc.data();
      const li = document.createElement("li");
      const dt = d.datetime ?? new Date(d.timestamp).toISOString();
      li.textContent = `${dt} | O:${d.open} H:${d.high} L:${d.low} C:${d.close} Vol:${d.volume}`;
      barsList.appendChild(li);
    });
  } catch (err) {
    barsList.innerHTML = `<li class="err">Error: ${err.message}</li>`;
  }
}

// ===== Storage: simple upload =====
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    uploadStatusEl.textContent = "Choose a file first.";
    uploadStatusEl.className = "muted";
    return;
  }
  try {
    const uid = auth.currentUser?.uid ?? "anonymous";
    const storageRef = ref(storage, `uploads/${uid}/${file.name}`);
    await uploadBytes(storageRef, file);
    uploadStatusEl.textContent = `Uploaded ${file.name}`;
    uploadStatusEl.className = "ok";
  } catch (err) {
    uploadStatusEl.textContent = `Upload error: ${err.message}`;
    uploadStatusEl.className = "err";
  }
});
