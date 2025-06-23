package cmd

import (
	"os"

	"github.com/spf13/cobra"
)

// completionCmd represents the completion command
var completionCmd = &cobra.Command{
	Use:   "completion [bash|zsh|fish|powershell]",
	Short: "Generate shell completion script",
	Long: `Generate shell completion script for devops command.

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
	Run: func(cmd *cobra.Command, args []string) {
		switch args[0] {
		case "bash":
			cmd.Root().GenBashCompletion(os.Stdout)
		case "zsh":
			cmd.Root().GenZshCompletion(os.Stdout)
		case "fish":
			cmd.Root().GenFishCompletion(os.Stdout, true)
		case "powershell":
			cmd.Root().GenPowerShellCompletionWithDesc(os.Stdout)
		}
	},
}