import { useState, useEffect } from 'react';

/**
 * Custom hook for persisting state in localStorage
 * @param {string} key - The localStorage key
 * @param {*} defaultValue - Default value if nothing is stored
 * @returns {[*, function]} - [value, setValue] tuple
 */
export const usePersistedState = (key, defaultValue) => {
  const [state, setState] = useState(() => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.warn(`Failed to parse localStorage item "${key}":`, error);
      localStorage.removeItem(key);
      return defaultValue;
    }
  });

  const setValue = (value) => {
    try {
      setState(value);
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Failed to save to localStorage "${key}":`, error);
    }
  };

  return [state, setValue];
};

/**
 * Hook specifically for persisting selected project IDs
 * @returns {[string[], function]} - [selectedProjectIds, setSelectedProjectIds] tuple
 */
export const usePersistedProjectSelection = () => {
  return usePersistedState('chatInterface_selectedProjects', []);
};
