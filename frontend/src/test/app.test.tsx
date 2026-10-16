import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { Header } from '../components/Header';
import type { Project } from '../types';

describe('Header component', () => {
  const mockProjects: Project[] = [
    {
      id: 'proj-1',
      name: 'Project Alpha',
      description: 'First test project',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      dataset_count: 2,
      architecture_count: 1,
      experiment_count: 3
    }
  ];

  it('renders application branding and workspace navigation tabs', () => {
    const handleSelectWorkspace = vi.fn();
    const handleSelectProject = vi.fn();
    const handleNewProject = vi.fn();

    render(
      <Header
        projects={mockProjects}
        activeProject={mockProjects[0]}
        onSelectProject={handleSelectProject}
        onNewProject={handleNewProject}
        activeWorkspace="overview"
        onSelectWorkspace={handleSelectWorkspace}
        backendHealthy={true}
      />
    );

    expect(screen.getByText('NeuralForge')).toBeInTheDocument();
    expect(screen.getByText('Network Builder')).toBeInTheDocument();
    expect(screen.getByText('Dataset Studio')).toBeInTheDocument();
    expect(screen.getByText('Training Lab')).toBeInTheDocument();
    expect(screen.getByText('Model Inspector')).toBeInTheDocument();
    expect(screen.getByText('Experiment Center')).toBeInTheDocument();
    expect(screen.getByText('● Core Ready')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Training Lab'));
    expect(handleSelectWorkspace).toHaveBeenCalledWith('training');
  });

  it('renders offline badge when backend is unreachable', () => {
    render(
      <Header
        projects={mockProjects}
        activeProject={mockProjects[0]}
        onSelectProject={() => {}}
        onNewProject={() => {}}
        activeWorkspace="overview"
        onSelectWorkspace={() => {}}
        backendHealthy={false}
      />
    );

    expect(screen.getByText('● Core Offline')).toBeInTheDocument();
  });
});
