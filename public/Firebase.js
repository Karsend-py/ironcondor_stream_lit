
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";
import { getStorage } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-storage.js";

const firebaseConfig = {
  apiKey: "AIzaSyD_QGDTdAw9MZtqGLzwoyIhC9TYxdsOw94",
  authDomain: "glimmerglass-fa385.firebaseapp.com",
  projectId: "glimmerglass-fa385",
  storageBucket: "glimmerglass-fa385.firebasestorage.app",
  messagingSenderId: "932325809892",
  appId: "1:932325809892:web:70ea2e3754ec42161796f1"
};

export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);
