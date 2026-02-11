import { CustomFilterDefinition } from '../types/customFilters';
import { getElectronAPI, isElectron } from './electron-api';

export async function getCustomFilters(): Promise<CustomFilterDefinition[]> {
  if (!isElectron()) {
    return [];
  }

  const api = getElectronAPI();
  const response = await api.customFilters.list();

  if (!response.success || !response.data) {
    return [];
  }

  return response.data;
}

export async function createCustomFilter(payload: {
  label: string;
  type: 'dropdown' | 'text';
  options?: string[];
}): Promise<CustomFilterDefinition | null> {
  if (!isElectron()) {
    return null;
  }

  const api = getElectronAPI();
  const response = await api.customFilters.create(payload);

  if (!response.success || !response.data) {
    return null;
  }

  return response.data;
}

export async function deleteCustomFilter(id: number): Promise<boolean> {
  if (!isElectron()) {
    return false;
  }

  const api = getElectronAPI();
  const response = await api.customFilters.delete(id);
  return response.success === true;
}

export async function updateCustomFilterOptions(
  id: number,
  options: string[]
): Promise<CustomFilterDefinition | null> {
  if (!isElectron()) {
    return null;
  }

  const api = getElectronAPI();
  const response = await api.customFilters.updateOptions(id, options);

  if (!response.success || !response.data) {
    return null;
  }

  return response.data;
}
