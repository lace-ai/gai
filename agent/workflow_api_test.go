package agent_test

import (
	"reflect"
	"testing"

	"github.com/lace-ai/gai/agent"
)

func TestWorkflowDoesNotExposeMutableLoop(t *testing.T) {
	if _, ok := reflect.TypeOf(agent.Workflow{}).FieldByName("Loop"); ok {
		t.Fatal("Workflow must not expose its mutable loop")
	}
}
