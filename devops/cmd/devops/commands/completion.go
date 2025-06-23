// Package commands provides shell completion support
package commands

import (
	"github.com/spf13/cobra"

	"github.com/raja-aiml/sematic-cache/devops/pkg/factory"
)

// CompletionCommand handles shell completion generation
type CompletionCommand struct {
	*BaseCommand
	cmd *cobra.Command
}

// NewCompletionCommand creates a new completion command
func NewCompletionCommand(factory *factory.Factory) *CompletionCommand {
	cc := &CompletionCommand{
		BaseCommand: NewBaseCommand(factory),
	}

	cc.cmd = &cobra.Command{
		Use:   "completion [bash|zsh|fish|powershell]",
		Short: "Generate shell completion script",
		Long: `Generate shell completion script for devops CLI.

To load completions:

Bash:
  $ source <(devops completion bash)
  # To load completions for each session, execute once:
  # Linux:
  $ devops completion bash > /etc/bash_completion.d/devops
  # macOS:
  $ devops completion bash > /usr/local/etc/bash_completion.d/devops

Zsh:
  $ source <(devops completion zsh)
  # To load completions for each session, execute once:
  $ devops completion zsh > "${fpath[1]}/_devops"

Fish:
  $ devops completion fish | source
  # To load completions for each session, execute once:
  $ devops completion fish > ~/.config/fish/completions/devops.fish

PowerShell:
  PS> devops completion powershell | Out-String | Invoke-Expression
  # To load completions for every new session, run:
  PS> devops completion powershell > devops.ps1
  # and source this file from your PowerShell profile.
`,
		DisableFlagsInUseLine: true,
		ValidArgs:             []string{"bash", "zsh", "fish", "powershell"},
		Args:                  cobra.ExactValidArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			switch args[0] {
			case "bash":
				return cmd.Root().GenBashCompletion(cmd.OutOrStdout())
			case "zsh":
				return cmd.Root().GenZshCompletion(cmd.OutOrStdout())
			case "fish":
				return cmd.Root().GenFishCompletion(cmd.OutOrStdout(), true)
			case "powershell":
				return cmd.Root().GenPowerShellCompletionWithDesc(cmd.OutOrStdout())
			}
			return nil
		},
	}

	return cc
}

// GetCommand returns the cobra command
func (cc *CompletionCommand) GetCommand() *cobra.Command {
	return cc.cmd
}
